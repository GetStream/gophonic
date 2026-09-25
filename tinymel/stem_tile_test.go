// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package tinymel

import (
	"math"
	"testing"
)

func TestTinyStemTile2x4ParityRangesAndAllocations(t *testing.T) {
	for _, inputLength := range []int{1, 2, 3, 4, 5, 11, 32} {
		conv := syntheticTinyConv(80, 192, 5, 2, 1, int64(inputLength)+101)
		outputLength := (inputLength+4-5)/2 + 1
		input := make([]uint8, inputLength*conv.inChannels)
		for i := range input {
			input[i] = uint8(i*73 + 19)
		}
		got := make([]float32, outputLength*conv.outChannels)
		want := make([]float32, len(got))
		scalar := make([]float32, len(got))
		for _, zero := range []uint8{0, 1, 127, 128, 254, 255} {
			for _, weightZero := range []uint8{0, 128, 255} {
				conv.weightZero = weightZero
				params := tinyQuantParams{scale: 0.03125, zero: zero}
				runTinyConvTile4Range(conv, input, params, want, inputLength, 0, outputLength)
				runTinyConvScalarRange(conv, input, params, scalar, inputLength, 0, outputLength)
				assertTinyStemTileEqual(t, want, scalar, inputLength, zero, weightZero, -2)
				runTinyStemTile2x4Range(conv, input, params, got, inputLength, 0, outputLength)
				assertTinyStemTileEqual(t, got, want, inputLength, zero, weightZero, -1)

				for cut := 0; cut <= outputLength; cut++ {
					clear(got)
					runTinyStemTile2x4Range(conv, input, params, got, inputLength, -3, cut)
					runTinyStemTile2x4Range(conv, input, params, got, inputLength, cut, outputLength+4)
					assertTinyStemTileEqual(t, got, want, inputLength, zero, weightZero, cut)
				}
			}
		}

		params := tinyQuantParams{scale: 0.015625, zero: 253}
		if allocs := testing.AllocsPerRun(5, func() {
			runTinyStemTile2x4Range(conv, input, params, got, inputLength, 0, outputLength)
		}); allocs != 0 {
			t.Fatalf("inputLength=%d: kernel allocated %g objects", inputLength, allocs)
		}
	}
}

func assertTinyStemTileEqual(t *testing.T, got, want []float32, inputLength int, zero, weightZero uint8, cut int) {
	t.Helper()
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("inputLength=%d zero=%d weightZero=%d cut=%d index=%d got=%g want=%g", inputLength, zero, weightZero, cut, i, got[i], want[i])
		}
	}
}

func TestTinyStemTile2x4FallsBackForOtherShapes(t *testing.T) {
	conv := syntheticTinyConv(80, 12, 3, 2, 1, 811)
	const inputLength = 13
	outputLength := (inputLength+2*(conv.kernel/2)-conv.kernel)/conv.stride + 1
	input := make([]uint8, inputLength*conv.inChannels)
	for i := range input {
		input[i] = uint8(i*43 + 7)
	}
	params := tinyQuantParams{scale: 0.0078125, zero: 255}
	got, want := make([]float32, outputLength*conv.outChannels), make([]float32, outputLength*conv.outChannels)
	runTinyStemTile2x4Range(conv, input, params, got, inputLength, 0, outputLength)
	runTinyConvTile4Range(conv, input, params, want, inputLength, 0, outputLength)
	assertTinyStemTileEqual(t, got, want, inputLength, params.zero, conv.weightZero, -1)
}

func BenchmarkTinyStemTile2x4(b *testing.B) {
	const inputLength = 800
	conv := syntheticTinyConv(80, 192, 5, 2, 1, 731)
	input := make([]uint8, inputLength*conv.inChannels)
	for i := range input {
		input[i] = uint8(i*37 + 91)
	}
	outputLength := (inputLength+4-5)/2 + 1
	output := make([]float32, outputLength*conv.outChannels)
	params := tinyQuantParams{scale: 0.015625, zero: 121}
	for _, kernel := range []struct {
		name string
		run  func(tinyConv1D, []uint8, tinyQuantParams, []float32, int, int, int) int
	}{
		{"tile4", runTinyConvTile4Range},
		{"tile2x4", runTinyStemTile2x4Range},
	} {
		b.Run(kernel.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				kernel.run(conv, input, params, output, inputLength, 0, outputLength)
			}
		})
	}
}
