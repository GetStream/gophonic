// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gophonic

import (
	"math"
	"testing"
)

func TestTinyDepthwiseSIMDParity(t *testing.T) {
	for _, channels := range []int{16, 32, 192, 19} {
		for _, length := range []int{1, 2, 17} {
			for _, stride := range []int{1, 2} {
				conv := syntheticTinyConv(channels, channels, 5, stride, channels, 593)
				packed := packTinyDepthwiseWeights(conv)
				input := make([]uint8, length*channels)
				for i := range input {
					input[i] = uint8(i*73 + 19)
				}
				outLen := (length-1)/stride + 1
				got, want := make([]float32, outLen*channels), make([]float32, outLen*channels)
				for _, zero := range []uint8{0, 127, 128, 255} {
					for _, wz := range []uint8{0, 128, 255} {
						conv.weightZero = wz
						params := tinyQuantParams{scale: 0.017123, zero: zero}
						runTinyConvScalar(conv, input, params, want, length)
						runTinyDepthwiseSIMDRange(conv, packed, input, params, got, length, -3, outLen/2)
						runTinyDepthwiseSIMDRange(conv, packed, input, params, got, length, outLen/2, outLen+5)
						for i := range got {
							if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
								t.Fatalf("channels=%d length=%d stride=%d zero=%d weightZero=%d output[%d]=%g want=%g", channels, length, stride, zero, wz, i, got[i], want[i])
							}
						}
					}
				}
				params := tinyQuantParams{scale: 0.015625, zero: 121}
				if allocs := testing.AllocsPerRun(5, func() { runTinyDepthwiseSIMDRange(conv, packed, input, params, got, length, 0, outLen) }); allocs != 0 {
					t.Fatalf("depthwise allocated %g objects", allocs)
				}
			}
		}
	}
}

func BenchmarkTinyDepthwiseSIMD(b *testing.B) {
	conv := syntheticTinyConv(192, 192, 5, 2, 192, 593)
	packed := packTinyDepthwiseWeights(conv)
	input := make([]uint8, 400*192)
	for i := range input {
		input[i] = uint8(i*37 + 91)
	}
	output := make([]float32, 200*192)
	params := tinyQuantParams{scale: 0.015625, zero: 121}
	b.Run("baseline", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			runTinyConvSIMDRange(conv, input, params, output, 400, 0, 200)
		}
	})
	b.Run("channel16", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			runTinyDepthwiseSIMDRange(conv, packed, input, params, output, 400, 0, 200)
		}
	})
}
