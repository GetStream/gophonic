// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"fmt"
	"testing"
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
