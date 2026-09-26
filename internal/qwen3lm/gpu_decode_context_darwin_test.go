// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"fmt"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

// BenchmarkOfficialGPUDecodeContext times one token of Qwen3-8B decoding
// after a context of each length: the step a reply takes per word. Weight
// bytes are the same at every length; the growth is attention reading the
// keys and values. It reports the effective bandwidth over weights and
// keys and values read.
func BenchmarkOfficialGPUDecodeContext(b *testing.B) {
	for _, format := range []string{WeightsGPUQ8, WeightsGPUQ4} {
		lm := loadOfficialLM(b, format)
		c := lm.weights.Config()
		kv, err := lm.eval.NewPrefixKV(gpuPositions)
		if err != nil {
			b.Fatal(err)
		}
		ids := make([]int, gpuPositions)
		for i := range ids {
			ids[i] = 1000 + (i*7919)%50000
		}
		dst := lm.rows(1)[0]
		for _, n := range []int{128, 512, 1024, 2000} {
			if err := lm.eval.HiddenLastExtendInto(kv, 0, ids[:n], dst, lm.ws); err != nil {
				b.Fatal(err)
			}
			b.Run(fmt.Sprintf("%s/ctx%d", format, n), func(b *testing.B) {
				for b.Loop() {
					if err := lm.eval.HiddenLastExtendInto(kv, n, ids[n:n+1], dst, lm.ws); err != nil {
						b.Fatal(err)
					}
				}
				kvBytes := float64(c.Layers * 2 * c.KVHeads * c.HeadDim * 4 * n)
				perToken := b.Elapsed().Seconds() / float64(b.N)
				b.ReportMetric(perToken*1e3, "ms/token")
				b.ReportMetric((float64(len(lm.weights.gpu.cache.Payload()))+kvBytes)/perToken/1e9, "GB/s")
				b.ReportMetric(kvBytes/1e6, "MB-kv")
			})
		}
		lm.weights.Release()
	}
}

// TestOfficialGPUSplitAttention holds one token's attention split across the
// keys and merged to the single-threadgroup kernel, at context lengths past
// the split, on Qwen3-8B.
func TestOfficialGPUSplitAttention(t *testing.T) {
	lm := loadOfficialLM(t, WeightsGPUQ8)
	kv, err := lm.eval.NewPrefixKV(gpuPositions)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int, gpuPositions)
	for i := range ids {
		ids[i] = 1000 + (i*7919)%50000
	}
	defer func(min int) { attendChunkMin = min }(attendChunkMin)
	for _, n := range []int{300, 700, 1500, 2000} {
		if err := lm.eval.HiddenLastExtendInto(kv, 0, ids[:n], lm.rows(1)[0], lm.ws); err != nil {
			t.Fatal(err)
		}
		step := func(min int) []float32 {
			attendChunkMin = min
			out := lm.rows(1)[0]
			if err := lm.eval.HiddenLastExtendInto(kv, n, ids[n:n+1], out, lm.ws); err != nil {
				t.Fatal(err)
			}
			return out
		}
		whole, split := step(gpuPositions), step(0)
		cos, maxAbs := lmtest.VectorParity(split, whole)
		t.Logf("%d positions: cosine %.7f, max_abs %.3g", n, cos, maxAbs)
		if cos < 0.99999 {
			t.Errorf("%d positions: split attention diverges (cosine %.7f)", n, cos)
		}
	}
}
