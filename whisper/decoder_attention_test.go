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
		s := &DecoderScratch{query: query, context: make([]float32, TextState), scaledQuery: make([]float32, TextState), scores: make([]float32, TextHeads*AudioFrames)}
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
