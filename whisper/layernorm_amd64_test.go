// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whisper

import (
	"fmt"
	"math"
	"testing"
)

func BenchmarkAMD64LayerNorm(b *testing.B) {
	for _, n := range []int{384, 1024, 1536} {
		for _, variant := range []struct {
			name string
			fn   func([]float32, []float32, []float32, []float32)
		}{
			{"dispatched", layerNormRow},
			{"generic", layerNormRowGeneric},
		} {
			b.Run(fmt.Sprintf("%s/n%d", variant.name, n), func(b *testing.B) {
				src, dst := make([]float32, n), make([]float32, n)
				gamma, beta := make([]float32, n), make([]float32, n)
				for i := range src {
					src[i] = 50 + float32(math.Sin(float64(i)*0.19))*3
					gamma[i] = float32(math.Cos(float64(i) * 0.13))
					beta[i] = float32(i%17) * 0.01
				}
				b.ReportAllocs()
				for b.Loop() {
					variant.fn(src, dst, gamma, beta)
				}
				encoderBenchmarkSink = dst[n-1]
			})
		}
	}
}
