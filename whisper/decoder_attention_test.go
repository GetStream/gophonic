// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

func TestDecoderAttentionCacheLayoutAndWorkers(t *testing.T) {
	pool, err := whispergemm.NewExecutor(4)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	for _, frames := range []int{1, 7, 128, AudioFrames} {
		query := make([]float32, TextState)
		keys, values := make([]float32, frames*TextState), make([]float32, frames*TextState)
		for i := range query {
			query[i] = float32(i%31-15) / 16
		}
		for i := range keys {
			keys[i] = float32(i%37-18) / 19
			values[i] = float32(i%41-20) / 21
		}
		want := make([]float32, TextState)
		attentionInto(want, query, keys, values, frames, TextHeads, make([]float32, frames))
		scale := float32(math.Pow(float64(TextState/TextHeads), -0.25))
		for i := range keys {
			keys[i] *= scale
		}
		transposed := make([]float32, len(values))
		for frame := 0; frame < frames; frame++ {
			for d := 0; d < TextState; d++ {
				transposed[d*frames+frame] = values[frame*TextState+d]
			}
		}
		s := &DecoderScratch{dims: TinyENDims, query: query, context: make([]float32, TextState), scaledQuery: make([]float32, TextState), scores: make([]float32, TextHeads*AudioFrames)}
		if err := s.attend(keys, transposed, frames, frames); err != nil {
			t.Fatal(err)
		}
		serial := append([]float32(nil), s.context...)
		for i, got := range serial {
			if math.Abs(float64(got-want[i])) > 2e-5 {
				t.Fatalf("frames=%d output[%d]=%g want %g", frames, i, got, want[i])
			}
		}
		s.gemm = pool
		if allocations := testing.AllocsPerRun(3, func() {
			if err := s.attend(keys, transposed, frames, frames); err != nil {
				panic(err)
			}
		}); allocations != 0 {
			t.Fatalf("frames=%d attention allocated %g objects", frames, allocations)
		}
		for i, got := range s.context {
			if math.Float32bits(got) != math.Float32bits(serial[i]) {
				t.Fatalf("frames=%d worker output[%d]=%g want exact serial %g", frames, i, got, serial[i])
			}
		}
	}
}

// Compare the public attention dispatch with the previous head-parallel
// evaluation, including both sides of the small packed-head cutoff.
func TestPackedDecoderAttentionDispatchIsExact(t *testing.T) {
	pool, err := whispergemm.NewExecutor(4)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	for _, shape := range [][3]int{{6, 64, 128}, {6, 64, AudioFrames}, {8, 64, AudioFrames}, {6, 80, 129}} {
		heads, width, frames := shape[0], shape[1], shape[2]
		state := heads * width
		dims := TinyENDims
		dims.TextHeads, dims.TextState = heads, state
		s := &DecoderScratch{dims: dims, gemm: pool,
			query: make([]float32, state), scaledQuery: make([]float32, state),
			context: make([]float32, state), scores: make([]float32, heads*AudioFrames),
			layerKeys: make([]*whispergemm.PackedVector, heads), layerValues: make([]*whispergemm.PackedVector, heads)}
		for i := range s.query {
			s.query[i] = float32(i%31-15) / 17
		}
		keys, values := make([]float32, frames*width), make([]float32, width*frames)
		for head := range heads {
			for i := range keys {
				keys[i] = float32((i+head)%37-18) / 19
				values[i] = float32((i+3*head)%41-20) / 21
			}
			if s.layerKeys[head], err = whispergemm.NewPackedVector(keys, width, frames, width); err != nil {
				t.Fatal(err)
			}
			if s.layerValues[head], err = whispergemm.NewPackedVector(values, frames, width, frames); err != nil {
				t.Fatal(err)
			}
		}
		scale := float32(math.Pow(float64(width), -0.25))
		for i, value := range s.query {
			s.scaledQuery[i] = value * scale
		}
		want := make([]float32, state)
		op := &decoderAttentionOperation{keyVec: s.layerKeys, valueVec: s.layerValues,
			heads: heads, dst: want, scaledQuery: s.scaledQuery, scores: make([]float32, heads*AudioFrames), frames: frames}
		if err := pool.Rows(op, heads, 1); err != nil {
			t.Fatal(err)
		}
		for range 3 {
			if err := s.attend(nil, nil, frames, frames); err != nil {
				t.Fatal(err)
			}
			for i, v := range want {
				if math.Float32bits(s.context[i]) != math.Float32bits(v) {
					t.Fatalf("shape=%v output[%d]=%08x want %08x", shape, i, math.Float32bits(s.context[i]), math.Float32bits(v))
				}
			}
		}
		if n := testing.AllocsPerRun(1, func() {
			if err := s.attend(nil, nil, frames, frames); err != nil {
				panic(err)
			}
		}); n != 0 {
			t.Fatalf("shape=%v allocated %g objects", shape, n)
		}
	}
}
