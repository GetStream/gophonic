// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package smartturn runs Pipecat's Smart Turn v3.2 end-of-turn detector: a
// four-layer Whisper-style audio encoder with an attention-pooled classifier
// over the latest eight seconds of audio.
package smartturn

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/GetStream/gophonic/internal/mel"
	"github.com/GetStream/gophonic/speech"
)

const (
	BundleMagic     = "GOINFER1"
	modelVersion    = uint32(1)
	melCount        = mel.TurnBins
	frameCount      = mel.TurnFrames
	sequenceLength  = 400
	hiddenSize      = 384
	attentionHeads  = 6
	headSize        = hiddenSize / attentionHeads
	feedForwardSize = 1536
)

// Prediction is the model's end-of-turn decision.
type Prediction = speech.Prediction

type linear struct {
	w []float32 // row-major [output,input]
	b []float32
}

type encoderLayer struct {
	q, k, v, out           linear
	selfNormW, selfNormB   []float32
	fc1, fc2               linear
	finalNormW, finalNormB []float32
}

// Model is an immutable Smart Turn v3.2 inference model. It is safe to share
// across goroutines; each concurrent prediction needs its own Workspace.
type Model struct {
	conv1W, conv1B                   []float32
	conv2W, conv2B                   []float32
	positions                        []float32
	layers                           [4]encoderLayer
	encoderNormW, encoderNormB       []float32
	pool1, pool2                     linear
	classifier1                      linear
	classifierNormW, classifierNormB []float32
	classifier2, classifier3         linear
}

type tensorSpec struct {
	name  string
	shape []int
}

func expectedTensors() []tensorSpec {
	s := []tensorSpec{
		{"conv1.weight", []int{384, 80, 3}}, {"conv1.bias", []int{384}},
		{"conv2.weight", []int{384, 384, 3}}, {"conv2.bias", []int{384}},
		{"embed_positions.weight", []int{400, 384}},
	}
	for i := 0; i < 4; i++ {
		p := fmt.Sprintf("layers.%d.", i)
		for _, n := range []string{"q", "k", "v", "out"} {
			s = append(s, tensorSpec{p + n + ".weight", []int{384, 384}})
		}
		s = append(s,
			tensorSpec{p + "q.bias", []int{384}},
			tensorSpec{p + "v.bias", []int{384}},
			tensorSpec{p + "out.bias", []int{384}},
			tensorSpec{p + "self_norm.weight", []int{384}},
			tensorSpec{p + "self_norm.bias", []int{384}},
			tensorSpec{p + "fc1.weight", []int{1536, 384}},
			tensorSpec{p + "fc1.bias", []int{1536}},
			tensorSpec{p + "fc2.weight", []int{384, 1536}},
			tensorSpec{p + "fc2.bias", []int{384}},
			tensorSpec{p + "final_norm.weight", []int{384}},
			tensorSpec{p + "final_norm.bias", []int{384}},
		)
	}
	s = append(s,
		tensorSpec{"encoder_norm.weight", []int{384}}, tensorSpec{"encoder_norm.bias", []int{384}},
		tensorSpec{"pool1.weight", []int{256, 384}}, tensorSpec{"pool1.bias", []int{256}},
		tensorSpec{"pool2.weight", []int{1, 256}}, tensorSpec{"pool2.bias", []int{1}},
		tensorSpec{"classifier1.weight", []int{256, 384}}, tensorSpec{"classifier1.bias", []int{256}},
		tensorSpec{"classifier_norm.weight", []int{256}}, tensorSpec{"classifier_norm.bias", []int{256}},
		tensorSpec{"classifier2.weight", []int{64, 256}}, tensorSpec{"classifier2.bias", []int{64}},
		tensorSpec{"classifier3.weight", []int{1, 64}}, tensorSpec{"classifier3.bias", []int{1}},
	)
	return s
}

// Load opens a converted Smart Turn v3.2 weight bundle.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadWeights(f)
}

// ReadWeights reads a bundle produced by tools/onnx_to_gophonic.py.
func ReadWeights(r io.Reader) (*Model, error) {
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read model header: %w", err)
	}
	if string(magic[:]) != BundleMagic {
		return nil, fmt.Errorf("unsupported model bundle magic %q", string(magic[:]))
	}
	var version, count uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read model version: %w", err)
	}
	if version != modelVersion {
		return nil, fmt.Errorf("unsupported model bundle version %d", version)
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}
	specs := expectedTensors()
	if int(count) != len(specs) {
		return nil, fmt.Errorf("expected %d tensors, bundle contains %d", len(specs), count)
	}
	specByName := make(map[string]tensorSpec, len(specs))
	for _, spec := range specs {
		specByName[spec.name] = spec
	}
	tensors := make(map[string][]float32, len(specs))
	for ti := uint32(0); ti < count; ti++ {
		var nameLen uint16
		if err := binary.Read(r, binary.LittleEndian, &nameLen); err != nil {
			return nil, fmt.Errorf("tensor %d name length: %w", ti, err)
		}
		if nameLen == 0 || nameLen > 256 {
			return nil, fmt.Errorf("tensor %d has invalid name length %d", ti, nameLen)
		}
		nameBytes := make([]byte, int(nameLen))
		if _, err := io.ReadFull(r, nameBytes); err != nil {
			return nil, fmt.Errorf("tensor %d name: %w", ti, err)
		}
		name := string(nameBytes)
		spec, ok := specByName[name]
		if !ok {
			return nil, fmt.Errorf("unexpected tensor %q", name)
		}
		if _, ok := tensors[name]; ok {
			return nil, fmt.Errorf("duplicate tensor %q", name)
		}
		var rank uint8
		if err := binary.Read(r, binary.LittleEndian, &rank); err != nil {
			return nil, fmt.Errorf("tensor %q rank: %w", name, err)
		}
		if int(rank) != len(spec.shape) {
			return nil, fmt.Errorf("tensor %q has rank %d, expected %d", name, rank, len(spec.shape))
		}
		shape := make([]int, int(rank))
		length := 1
		for i := range shape {
			var dim uint32
			if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
				return nil, fmt.Errorf("tensor %q dimension %d: %w", name, i, err)
			}
			if dim == 0 || uint64(dim) > uint64(math.MaxInt)/uint64(length) {
				return nil, fmt.Errorf("tensor %q has invalid dimension %d", name, dim)
			}
			shape[i] = int(dim)
			length *= int(dim)
		}
		for i := range shape {
			if shape[i] != spec.shape[i] {
				return nil, fmt.Errorf("tensor %q has shape %v, expected %v", name, shape, spec.shape)
			}
		}
		var valueCount uint32
		if err := binary.Read(r, binary.LittleEndian, &valueCount); err != nil {
			return nil, fmt.Errorf("tensor %q value count: %w", name, err)
		}
		if int(valueCount) != length {
			return nil, fmt.Errorf("tensor %q shape has %d values, bundle says %d", name, length, valueCount)
		}
		raw := make([]byte, length*4)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, fmt.Errorf("tensor %q data: %w", name, err)
		}
		values := make([]float32, length)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		tensors[name] = values
	}
	var trailing [1]byte
	if n, err := r.Read(trailing[:]); err != io.EOF || n != 0 {
		return nil, errors.New("model bundle has trailing data")
	}
	return modelFromTensors(tensors), nil
}

func modelFromTensors(t map[string][]float32) *Model {
	pull := func(name string) []float32 { return t[name] }
	linearFrom := func(name string) linear { return linear{w: pull(name + ".weight"), b: pull(name + ".bias")} }
	m := &Model{
		conv1W: pull("conv1.weight"), conv1B: pull("conv1.bias"),
		conv2W: pull("conv2.weight"), conv2B: pull("conv2.bias"),
		positions:    pull("embed_positions.weight"),
		encoderNormW: pull("encoder_norm.weight"), encoderNormB: pull("encoder_norm.bias"),
		pool1: linearFrom("pool1"), pool2: linearFrom("pool2"),
		classifier1:     linearFrom("classifier1"),
		classifierNormW: pull("classifier_norm.weight"), classifierNormB: pull("classifier_norm.bias"),
		classifier2: linearFrom("classifier2"), classifier3: linearFrom("classifier3"),
	}
	for i := range m.layers {
		p := fmt.Sprintf("layers.%d.", i)
		m.layers[i] = encoderLayer{
			q:         linear{w: pull(p + "q.weight"), b: pull(p + "q.bias")},
			k:         linear{w: pull(p + "k.weight")},
			v:         linear{w: pull(p + "v.weight"), b: pull(p + "v.bias")},
			out:       linearFrom(p + "out"),
			selfNormW: pull(p + "self_norm.weight"), selfNormB: pull(p + "self_norm.bias"),
			fc1: linearFrom(p + "fc1"), fc2: linearFrom(p + "fc2"),
			finalNormW: pull(p + "final_norm.weight"), finalNormB: pull(p + "final_norm.bias"),
		}
	}
	return m
}
