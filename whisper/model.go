// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package whisper runs the official OpenAI Whisper tiny.en model on the CPU.
// A Model is immutable after loading; callers give each concurrent run its own scratch.
package whisper

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"

	"github.com/GetStream/gophonic/internal/arena"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

const (
	BundleMagic   = "WHISPER1"
	bundleVersion = uint32(1)
	// bundleVersionDims adds explicit model dimensions for larger checkpoints.
	bundleVersionDims = uint32(2)
	MelBins           = 80
	MelFrames         = 3000
	AudioFrames       = 1500
	AudioState        = 384
	AudioHeads        = 6
	AudioLayers       = 4
	TextContext       = 448
	TextState         = 384
	TextHeads         = 6
	TextLayers        = 4
	VocabSize         = 51864
)

// Dims describes the variable width and depth of an English Whisper model.
// Mel bins, audio frames, text context, and vocabulary are shared by all
// English checkpoints and remain package constants.
type Dims struct {
	AudioState, AudioHeads, AudioLayers int
	TextState, TextHeads, TextLayers    int
}

// TinyENDims are the dimensions of the official tiny.en checkpoint.
var TinyENDims = Dims{AudioState: AudioState, AudioHeads: AudioHeads, AudioLayers: AudioLayers,
	TextState: TextState, TextHeads: TextHeads, TextLayers: TextLayers}

func (d Dims) valid() bool {
	ok := func(state, heads, layers int) bool {
		return state > 0 && state <= 4096 && heads > 0 && state%heads == 0 && layers > 0 && layers <= 64
	}
	return ok(d.AudioState, d.AudioHeads, d.AudioLayers) && ok(d.TextState, d.TextHeads, d.TextLayers) &&
		d.AudioState == d.TextState
}

// Model owns validated FP32 weights for an English OpenAI Whisper model.
type Model struct {
	memory  *arena.Arena
	tensors map[string][]float32
	dims    Dims

	// Encoder packing is immutable and shared by every lane on this model.
	encoderMu      sync.Mutex
	encoderPacking *packedEncoderWeights

	// vectors caches immutable matrix-vector packings shared by every
	// decoder on this model. It is populated on first use.
	vectorMu sync.Mutex
	vectors  map[string]*whispergemm.PackedVector
}

// Close releases the model's mapped weights. Close every transcriber and
// stop all encoder/decoder calls first; the model and its scratch must not
// be used afterwards. Repeated Close calls are harmless.
func (m *Model) Close() error {
	if m == nil {
		return nil
	}
	if err := m.memory.Close(); err != nil {
		return err
	}
	m.memory = nil
	m.tensors, m.vectors, m.encoderPacking = nil, nil, nil
	return nil
}

// packedVector returns the shared packing of an N-by-K tensor, or nil when
// the platform has no accelerated matrix-vector path.
func (m *Model) packedVector(name string, rows, k int) (*whispergemm.PackedVector, error) {
	if !whispergemm.PackedVectorAccelerated() {
		return nil, nil
	}
	m.vectorMu.Lock()
	defer m.vectorMu.Unlock()
	if p := m.vectors[name]; p != nil {
		return p, nil
	}
	p, err := whispergemm.NewPackedVector(m.tensor(name), k, rows, k)
	if err != nil {
		return nil, err
	}
	if m.vectors == nil {
		m.vectors = make(map[string]*whispergemm.PackedVector)
	}
	m.vectors[name] = p
	return p, nil
}

func (m *Model) tensor(name string) []float32 { return m.tensors[name] }

// Dims returns the model's dimensions.
func (m *Model) Dims() Dims { return m.dims }

type tensorSpec struct {
	name  string
	shape []int
}

func expectedTensors(d Dims) []tensorSpec {
	a, t := d.AudioState, d.TextState
	s := []tensorSpec{
		{"encoder.conv1.weight", []int{a, MelBins, 3}},
		{"encoder.conv1.bias", []int{a}},
		{"encoder.conv2.weight", []int{a, a, 3}},
		{"encoder.conv2.bias", []int{a}},
		{"encoder.positional_embedding", []int{AudioFrames, a}},
		{"encoder.ln_post.weight", []int{a}},
		{"encoder.ln_post.bias", []int{a}},
		{"decoder.token_embedding.weight", []int{VocabSize, t}},
		{"decoder.positional_embedding", []int{TextContext, t}},
		{"decoder.ln.weight", []int{t}},
		{"decoder.ln.bias", []int{t}},
	}
	addLinear := func(prefix string, in, out int, bias bool) {
		s = append(s, tensorSpec{prefix + ".weight", []int{out, in}})
		if bias {
			s = append(s, tensorSpec{prefix + ".bias", []int{out}})
		}
	}
	for _, path := range []string{"encoder", "decoder"} {
		state, layers := a, d.AudioLayers
		if path == "decoder" {
			state, layers = t, d.TextLayers
		}
		addNorm := func(prefix string) {
			s = append(s, tensorSpec{prefix + ".weight", []int{state}}, tensorSpec{prefix + ".bias", []int{state}})
		}
		for i := 0; i < layers; i++ {
			p := fmt.Sprintf("%s.blocks.%d.", path, i)
			for _, attention := range []string{"attn", "cross_attn"} {
				if attention == "cross_attn" && path == "encoder" {
					continue
				}
				q := p + attention + "."
				addLinear(q+"query", state, state, true)
				addLinear(q+"key", state, state, false)
				addLinear(q+"value", state, state, true)
				addLinear(q+"out", state, state, true)
				addNorm(p + attention + "_ln")
			}
			addLinear(p+"mlp.0", state, 4*state, true)
			addLinear(p+"mlp.2", 4*state, state, true)
			addNorm(p + "mlp_ln")
		}
	}
	return s
}

// Load opens a converted tiny.en weight bundle. Conversion is an offline step;
// inference itself uses only Go and the standard library.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadWeights(f)
}

// ReadWeights validates all names, shapes, values, and trailing bytes.
func ReadWeights(r io.Reader) (*Model, error) {
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read Whisper header: %w", err)
	}
	if string(magic[:]) != BundleMagic {
		return nil, errors.New("invalid Whisper bundle magic")
	}
	var version, count uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != bundleVersion && version != bundleVersionDims {
		return nil, fmt.Errorf("unsupported Whisper bundle version %d", version)
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	hash := sha256.New()
	payload := io.TeeReader(r, hash)
	dims := TinyENDims
	if version == bundleVersionDims {
		// Version 2 records the dimensions inside the checksummed payload.
		var raw [6]uint32
		if err := binary.Read(payload, binary.LittleEndian, &raw); err != nil {
			return nil, err
		}
		dims = Dims{int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]), int(raw[4]), int(raw[5])}
		if !dims.valid() {
			return nil, fmt.Errorf("invalid Whisper dimensions %+v", dims)
		}
	}
	specs := expectedTensors(dims)
	if int(count) != len(specs) {
		return nil, fmt.Errorf("Whisper bundle has %d tensors; need %d", count, len(specs))
	}
	want := make(map[string][]int, len(specs))
	for _, s := range specs {
		want[s.name] = s.shape
	}
	lengths := make([]int, len(specs))
	for i, s := range specs {
		n := 1
		for _, d := range s.shape {
			if d <= 0 || n > int(^uint(0)>>1)/d {
				return nil, arena.ErrSize
			}
			n *= d
		}
		lengths[i] = n
	}
	memory, err := arena.New(lengths...)
	if err != nil {
		return nil, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = memory.Close()
		}
		runtime.KeepAlive(memory)
	}()
	// Bound transient read storage even for the large vocabulary tensor.
	buffer := make([]byte, 256<<10)
	tensors := make(map[string][]float32, len(specs))
	for range count {
		var n uint16
		if err := binary.Read(payload, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		if n == 0 || n > 256 {
			return nil, errors.New("invalid Whisper tensor name length")
		}
		buf := make([]byte, n)
		if _, err := io.ReadFull(payload, buf); err != nil {
			return nil, err
		}
		name := string(buf)
		shape, ok := want[name]
		if !ok {
			return nil, fmt.Errorf("unexpected Whisper tensor %q", name)
		}
		if _, duplicate := tensors[name]; duplicate {
			return nil, fmt.Errorf("duplicate Whisper tensor %q", name)
		}
		var rank uint8
		if err := binary.Read(payload, binary.LittleEndian, &rank); err != nil {
			return nil, err
		}
		if int(rank) != len(shape) {
			return nil, fmt.Errorf("tensor %q has wrong rank", name)
		}
		length := 1
		for _, expected := range shape {
			var dim uint32
			if err := binary.Read(payload, binary.LittleEndian, &dim); err != nil {
				return nil, err
			}
			if int(dim) != expected {
				return nil, fmt.Errorf("tensor %q has wrong dimension %d; want %d", name, dim, expected)
			}
			length *= expected
		}
		var values uint32
		if err := binary.Read(payload, binary.LittleEndian, &values); err != nil {
			return nil, err
		}
		if int(values) != length {
			return nil, fmt.Errorf("tensor %q has wrong element count", name)
		}
		v := memory.Take(length)
		for first := 0; first < length; {
			count := min(length-first, len(buffer)/4)
			data := buffer[:count*4]
			if _, err := io.ReadFull(payload, data); err != nil {
				return nil, err
			}
			if err := decodeFiniteFloat32(data, v[first:first+count]); err != nil {
				return nil, fmt.Errorf("tensor %q: %w", name, err)
			}
			first += count
		}
		tensors[name] = v
	}
	var digest [sha256.Size]byte
	if _, err := io.ReadFull(r, digest[:]); err != nil {
		return nil, fmt.Errorf("read Whisper bundle checksum: %w", err)
	}
	if !slices.Equal(digest[:], hash.Sum(nil)) {
		return nil, errors.New("Whisper bundle checksum mismatch")
	}
	var trailing [1]byte
	if n, err := r.Read(trailing[:]); n != 0 || err != io.EOF {
		return nil, errors.New("Whisper bundle has trailing data")
	}
	committed = true
	return &Model{tensors: tensors, dims: dims, memory: memory}, nil
}

func decodeFiniteFloat32(data []byte, dst []float32) error {
	for i := range dst {
		value := math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("non-finite value at index %d", i)
		}
		dst[i] = value
	}
	return nil
}
