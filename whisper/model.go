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
	"slices"
)

const (
	bundleMagic   = "WHISPER1"
	bundleVersion = uint32(1)
	MelBins       = 80
	MelFrames     = 3000
	AudioFrames   = 1500
	AudioState    = 384
	AudioHeads    = 6
	AudioLayers   = 4
	TextContext   = 448
	TextState     = 384
	TextHeads     = 6
	TextLayers    = 4
	VocabSize     = 51864
)

// Model owns validated FP32 weights for OpenAI Whisper tiny.en.
type Model struct{ tensors map[string][]float32 }

func (m *Model) tensor(name string) []float32 { return m.tensors[name] }

type tensorSpec struct {
	name  string
	shape []int
}

func expectedTensors() []tensorSpec {
	s := []tensorSpec{
		{"encoder.conv1.weight", []int{384, 80, 3}},
		{"encoder.conv1.bias", []int{384}},
		{"encoder.conv2.weight", []int{384, 384, 3}},
		{"encoder.conv2.bias", []int{384}},
		{"encoder.positional_embedding", []int{1500, 384}},
		{"encoder.ln_post.weight", []int{384}},
		{"encoder.ln_post.bias", []int{384}},
		{"decoder.token_embedding.weight", []int{51864, 384}},
		{"decoder.positional_embedding", []int{448, 384}},
		{"decoder.ln.weight", []int{384}},
		{"decoder.ln.bias", []int{384}},
	}
	addLinear := func(prefix string, in, out int, bias bool) {
		s = append(s, tensorSpec{prefix + ".weight", []int{out, in}})
		if bias {
			s = append(s, tensorSpec{prefix + ".bias", []int{out}})
		}
	}
	addNorm := func(prefix string) {
		s = append(s, tensorSpec{prefix + ".weight", []int{384}}, tensorSpec{prefix + ".bias", []int{384}})
	}
	for _, path := range []string{"encoder", "decoder"} {
		for i := 0; i < 4; i++ {
			p := fmt.Sprintf("%s.blocks.%d.", path, i)
			for _, a := range []string{"attn", "cross_attn"} {
				if a == "cross_attn" && path == "encoder" {
					continue
				}
				q := p + a + "."
				addLinear(q+"query", 384, 384, true)
				addLinear(q+"key", 384, 384, false)
				addLinear(q+"value", 384, 384, true)
				addLinear(q+"out", 384, 384, true)
				addNorm(p + a + "_ln")
			}
			addLinear(p+"mlp.0", 384, 1536, true)
			addLinear(p+"mlp.2", 1536, 384, true)
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
	if string(magic[:]) != bundleMagic {
		return nil, errors.New("invalid Whisper bundle magic")
	}
	var version, count uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != bundleVersion {
		return nil, fmt.Errorf("unsupported Whisper bundle version %d", version)
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	specs := expectedTensors()
	if int(count) != len(specs) {
		return nil, fmt.Errorf("Whisper bundle has %d tensors; need %d", count, len(specs))
	}
	want := make(map[string][]int, len(specs))
	for _, s := range specs {
		want[s.name] = s.shape
	}
	tensors := make(map[string][]float32, len(specs))
	hash := sha256.New()
	payload := io.TeeReader(r, hash)
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
		data := make([]byte, length*4)
		if _, err := io.ReadFull(payload, data); err != nil {
			return nil, err
		}
		v := make([]float32, length)
		if err := decodeFiniteFloat32(data, v); err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
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
	return &Model{tensors: tensors}, nil
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
