// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gofloor

import (
	"math"
	"testing"
)

func TestTinyConvTile4Parity(t *testing.T) {
	for _, spec := range []struct{ in, out, kernel, stride, groups, length int }{
		{80, 192, 5, 2, 1, 31},
		{192, 192, 1, 1, 1, 37},
		{16, 4, 5, 2, 1, 1},
		{32, 8, 3, 1, 1, 3},
		{19, 7, 3, 2, 1, 23},
		{32, 8, 3, 1, 2, 17},
		{192, 192, 5, 2, 192, 29},
	} {
		conv := syntheticTinyConv(spec.in, spec.out, spec.kernel, spec.stride, spec.groups, 471)
		input := make([]uint8, spec.in*spec.length)
		for i := range input {
			input[i] = uint8(i*97 + 31)
		}
		length := (spec.length+2*(spec.kernel/2)-spec.kernel)/spec.stride + 1
		got, want := make([]float32, length*spec.out), make([]float32, length*spec.out)
		for _, zero := range []uint8{0, 1, 127, 128, 254, 255} {
			for _, wz := range []uint8{0, 128, 255} {
				conv.weightZero = wz
				params := tinyQuantParams{scale: 0.03125, zero: zero}
				runTinyConvScalar(conv, input, params, want, spec.length)
				cut := length / 2
				runTinyConvTile4Range(conv, input, params, got, spec.length, -3, cut)
				runTinyConvTile4Range(conv, input, params, got, spec.length, cut, length+5)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("shape=%+v zero=%d weightZero=%d index=%d got=%g want=%g", spec, zero, wz, i, got[i], want[i])
					}
				}
			}
		}
		params := tinyQuantParams{scale: 0.015625, zero: 253}
		if allocs := testing.AllocsPerRun(5, func() {
			runTinyConvTile4Range(conv, input, params, got, spec.length, 0, length)
		}); allocs != 0 {
			t.Fatalf("tile4 allocated %g objects", allocs)
		}
	}
}

func BenchmarkTinyConvTile4(b *testing.B) {
	for _, spec := range []struct {
		name                            string
		in, out, kernel, stride, length int
	}{
		{"stem", 80, 192, 5, 2, 800},
		{"pointwise400", 192, 192, 1, 1, 400},
		{"pointwise200", 192, 192, 1, 1, 200},
		{"pointwise100", 192, 192, 1, 1, 100},
	} {
		conv := syntheticTinyConv(spec.in, spec.out, spec.kernel, spec.stride, 1, 731)
		input := make([]uint8, spec.in*spec.length)
		for i := range input {
			input[i] = uint8(i*37 + 91)
		}
		length := (spec.length+2*(spec.kernel/2)-spec.kernel)/spec.stride + 1
		output := make([]float32, length*spec.out)
		params := tinyQuantParams{scale: 0.015625, zero: 121}
		for _, kernel := range []struct {
			name string
			run  func(tinyConv1D, []uint8, tinyQuantParams, []float32, int, int, int) int
		}{
			{"baseline", runTinyConvSIMDRange},
			{"tile4", runTinyConvTile4Range},
		} {
			b.Run(spec.name+"/"+kernel.name, func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					kernel.run(conv, input, params, output, spec.length, 0, length)
				}
			})
		}
	}
}
