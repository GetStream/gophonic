// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package safetensors reads Hugging Face safetensors checkpoints, single
// file or sharded, without mapping them: tensors are read on demand into
// caller storage, so a loader can convert and pack one tensor at a time.
package safetensors

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/thesyncim/vibejson"
)

// Tensor locates one tensor in a checkpoint shard.
type Tensor struct {
	file   *os.File
	DType  string // "BF16", "F16", or "F32"
	Shape  []int
	offset int64
	size   int64
}

// Checkpoint is an open safetensors checkpoint directory.
type Checkpoint struct {
	dir     string
	files   []*os.File
	tensors map[string]Tensor
}

// Open reads the tensor index of dir: model.safetensors, or the shards
// listed by model.safetensors.index.json.
func Open(dir string) (*Checkpoint, error) {
	if !LittleEndian() {
		return nil, errors.New("safetensors: loading requires a little-endian CPU")
	}
	names := []string{"model.safetensors"}
	if raw, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := vibejson.Unmarshal(raw, &index); err != nil {
			return nil, fmt.Errorf("safetensors: parse index: %w", err)
		}
		seen := map[string]bool{}
		names = names[:0]
		for _, file := range index.WeightMap {
			if !seen[file] {
				seen[file] = true
				names = append(names, file)
			}
		}
	}
	c := &Checkpoint{dir: dir, tensors: map[string]Tensor{}}
	for _, name := range names {
		if filepath.Base(name) != name {
			c.Close()
			return nil, fmt.Errorf("safetensors: shard %q is not a plain file name", name)
		}
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			c.Close()
			return nil, fmt.Errorf("safetensors: open weights: %w", err)
		}
		c.files = append(c.files, f)
		if err := c.readHeader(f); err != nil {
			c.Close()
			return nil, fmt.Errorf("safetensors: %s: %w", name, err)
		}
	}
	return c, nil
}

func (c *Checkpoint) readHeader(f *os.File) error {
	info, err := f.Stat()
	if err != nil {
		return err
	}
	var lenBuf [8]byte
	if _, err := f.ReadAt(lenBuf[:], 0); err != nil {
		return fmt.Errorf("read header length: %w", err)
	}
	n := binary.LittleEndian.Uint64(lenBuf[:])
	if n == 0 || n > 100<<20 || int64(n)+8 > info.Size() {
		return errors.New("invalid safetensors header length")
	}
	header := make([]byte, n)
	if _, err := f.ReadAt(header, 8); err != nil {
		return fmt.Errorf("read header: %w", err)
	}
	// Safetensors pads the header with trailing spaces.
	header = bytes.TrimRight(header, " ")
	// __metadata__ decodes into an empty entry: its fields are unknown here.
	var entries map[string]struct {
		DType   string  `json:"dtype"`
		Shape   []int   `json:"shape"`
		Offsets []int64 `json:"data_offsets"`
	}
	if err := vibejson.Unmarshal(header, &entries); err != nil {
		return fmt.Errorf("parse header: %w", err)
	}
	base := int64(8 + n)
	for name, e := range entries {
		if name == "__metadata__" {
			continue
		}
		if len(e.Offsets) != 2 || e.Offsets[0] < 0 || e.Offsets[1] < e.Offsets[0] || base+e.Offsets[1] > info.Size() {
			return fmt.Errorf("tensor %s has invalid data offsets", name)
		}
		c.tensors[name] = Tensor{file: f, DType: e.DType, Shape: e.Shape, offset: base + e.Offsets[0], size: e.Offsets[1] - e.Offsets[0]}
	}
	return nil
}

// Dir returns the checkpoint directory.
func (c *Checkpoint) Dir() string { return c.dir }

// Close closes the shards. Tensors of a closed checkpoint cannot be read.
func (c *Checkpoint) Close() {
	for _, f := range c.files {
		_ = f.Close()
	}
	c.files = nil
}

// Has reports whether the checkpoint holds a tensor called name.
func (c *Checkpoint) Has(name string) bool {
	_, ok := c.tensors[name]
	return ok
}

// Lookup returns tensor name after checking that its shape is shape and its
// dtype is BF16, F16, or F32.
func (c *Checkpoint) Lookup(name string, shape ...int) (Tensor, error) {
	t, ok := c.tensors[name]
	if !ok {
		return t, fmt.Errorf("checkpoint is missing %s", name)
	}
	count := int64(1)
	for _, d := range shape {
		count *= int64(d)
	}
	if len(t.Shape) != len(shape) {
		return t, fmt.Errorf("%s has shape %v, want %v", name, t.Shape, shape)
	}
	for i := range shape {
		if t.Shape[i] != shape[i] {
			return t, fmt.Errorf("%s has shape %v, want %v", name, t.Shape, shape)
		}
	}
	width := map[string]int64{"BF16": 2, "F16": 2, "F32": 4}[t.DType]
	if width == 0 || t.size != count*width {
		return t, fmt.Errorf("%s has dtype %s and %d bytes for shape %v", name, t.DType, t.size, shape)
	}
	return t, nil
}

// BF16 reads a BF16 tensor's raw bits into new storage.
func (c *Checkpoint) BF16(name string, shape ...int) ([]uint16, error) {
	t, err := c.Lookup(name, shape...)
	if err != nil {
		return nil, err
	}
	if t.DType != "BF16" {
		return nil, fmt.Errorf("%s is %s; the loader expects the official BF16 checkpoint", name, t.DType)
	}
	out := make([]uint16, t.size/2)
	return out, t.ReadBits(out, 0)
}

// ReadBits reads len(dst) 16-bit elements starting at element offset of a
// BF16 or F16 tensor.
func (t Tensor) ReadBits(dst []uint16, offset int64) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(dst))), 2*len(dst))
	if offset < 0 || 2*(offset+int64(len(dst))) > t.size {
		return errors.New("read outside the tensor")
	}
	if _, err := t.file.ReadAt(b, t.offset+2*offset); err != nil {
		return fmt.Errorf("read tensor: %w", err)
	}
	return nil
}

// Float32 reads a BF16, F16, or F32 tensor with shape shape as FP32 and
// rejects non-finite values.
func (c *Checkpoint) Float32(name string, shape ...int) ([]float32, error) {
	t, err := c.Lookup(name, shape...)
	if err != nil {
		return nil, err
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	out := make([]float32, n)
	switch t.DType {
	case "F32":
		b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(out))), 4*n)
		if _, err := t.file.ReadAt(b, t.offset); err != nil {
			return nil, fmt.Errorf("read %s: %w", name, err)
		}
	default:
		raw := make([]uint16, n)
		if err := t.ReadBits(raw, 0); err != nil {
			return nil, err
		}
		for i, b := range raw {
			if t.DType == "BF16" {
				out[i] = math.Float32frombits(uint32(b) << 16)
			} else {
				out[i] = F16ToF32(b)
			}
		}
	}
	for _, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("%s has a non-finite value", name)
		}
	}
	return out, nil
}

// F16ToF32 converts IEEE half-precision bits to float32.
func F16ToF32(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	exp, mant := int(h>>10&0x1f), float64(h&0x3ff)
	switch exp {
	case 0:
		return sign * float32(math.Ldexp(mant, -24))
	case 31:
		if mant != 0 {
			return float32(math.NaN())
		}
		return sign * float32(math.Inf(1))
	}
	return sign * float32(math.Ldexp(1+mant/1024, exp-15))
}

// LittleEndian reports whether the CPU stores integers little-endian, as
// safetensors does.
func LittleEndian() bool {
	v := uint16(1)
	return *(*byte)(unsafe.Pointer(&v)) == 1
}
