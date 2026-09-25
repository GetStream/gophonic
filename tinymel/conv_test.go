// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package tinymel

import (
	"math"
	"testing"
)

func TestTinyConvSIMDParityAndAllocations(t *testing.T) {
	cases := []struct {
		name string
		conv tinyConv1D
		len  int
	}{
		{name: "stem", conv: syntheticTinyConv(80, 192, 5, 2, 1, 17), len: 31},
		{name: "depthwise", conv: syntheticTinyConv(192, 192, 5, 2, 192, 21), len: 29},
		{name: "pointwise", conv: syntheticTinyConv(192, 192, 1, 1, 1, 93), len: 37},
		{name: "tail", conv: syntheticTinyConv(19, 7, 3, 2, 1, 111), len: 23},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]uint8, tc.len*tc.conv.inChannels)
			for i := range input {
				input[i] = uint8((i*73 + 19) & 255)
			}
			params := tinyQuantParams{scale: 0.03125, zero: 203}
			wantLen := (tc.len+2*(tc.conv.kernel/2)-tc.conv.kernel)/tc.conv.stride + 1
			got := make([]float32, wantLen*tc.conv.outChannels)
			want := make([]float32, len(got))
			gotLen := runTinyConvSIMD(tc.conv, input, params, got, tc.len)
			wantLenGot := runTinyConvScalar(tc.conv, input, params, want, tc.len)
			if gotLen != wantLen || wantLenGot != wantLen {
				t.Fatalf("output lengths SIMD=%d scalar=%d, want %d", gotLen, wantLenGot, wantLen)
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("output[%d] = %.9g (0x%08x), want %.9g (0x%08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
				}
			}

			rangeGot := make([]float32, len(got))
			cut := wantLen / 2
			if n := runTinyConvSIMDRange(tc.conv, input, params, rangeGot, tc.len, 0, cut); n != wantLen {
				t.Fatalf("first SIMD range reports output length %d, want %d", n, wantLen)
			}
			if n := runTinyConvSIMDRange(tc.conv, input, params, rangeGot, tc.len, cut, wantLen); n != wantLen {
				t.Fatalf("second SIMD range reports output length %d, want %d", n, wantLen)
			}
			for i := range rangeGot {
				if math.Float32bits(rangeGot[i]) != math.Float32bits(want[i]) {
					t.Fatalf("split SIMD output[%d] = %.9g, want %.9g", i, rangeGot[i], want[i])
				}
			}

			rangeGot = make([]float32, len(got))
			runTinyConvScalarRange(tc.conv, input, params, rangeGot, tc.len, 0, cut)
			runTinyConvScalarRange(tc.conv, input, params, rangeGot, tc.len, cut, wantLen)
			for i := range rangeGot {
				if math.Float32bits(rangeGot[i]) != math.Float32bits(want[i]) {
					t.Fatalf("split scalar output[%d] = %.9g, want %.9g", i, rangeGot[i], want[i])
				}
			}

			dispatchGot := make([]float32, len(got))
			runTinyConvRange(tc.conv, input, params, dispatchGot, tc.len, 0, cut)
			runTinyConvRange(tc.conv, input, params, dispatchGot, tc.len, cut, wantLen)
			for i := range dispatchGot {
				if math.Float32bits(dispatchGot[i]) != math.Float32bits(want[i]) {
					t.Fatalf("dispatched range output[%d] = %.9g, want %.9g", i, dispatchGot[i], want[i])
				}
			}

			allocs := testing.AllocsPerRun(20, func() {
				runTinyConvSIMD(tc.conv, input, params, got, tc.len)
			})
			if allocs != 0 {
				t.Fatalf("runTinyConvSIMD allocated %.1f objects per call", allocs)
			}
			allocs = testing.AllocsPerRun(20, func() {
				runTinyConvSIMDRange(tc.conv, input, params, got, tc.len, 3, wantLen-2)
			})
			if allocs != 0 {
				t.Fatalf("runTinyConvSIMDRange allocated %.1f objects per call", allocs)
			}
			allocs = testing.AllocsPerRun(20, func() {
				runTinyConvScalarRange(tc.conv, input, params, got, tc.len, 3, wantLen-2)
			})
			if allocs != 0 {
				t.Fatalf("runTinyConvScalarRange allocated %.1f objects per call", allocs)
			}
		})
	}
}

func BenchmarkTinyConvSIMDKernels(b *testing.B) {
	bench := []struct {
		name string
		conv tinyConv1D
		len  int
	}{
		{name: "stem", conv: syntheticTinyConv(80, 192, 5, 2, 1, 17), len: 800},
		{name: "pointwise", conv: syntheticTinyConv(192, 192, 1, 1, 1, 93), len: 400},
		{name: "depthwise", conv: syntheticTinyConv(192, 192, 5, 1, 192, 47), len: 200},
	}
	for _, tc := range bench {
		input := make([]uint8, tc.len*tc.conv.inChannels)
		for i := range input {
			input[i] = uint8(i*37 + 91)
		}
		outLen := (tc.len+2*(tc.conv.kernel/2)-tc.conv.kernel)/tc.conv.stride + 1
		output := make([]float32, outLen*tc.conv.outChannels)
		b.Run(tc.name+"/simd", func(b *testing.B) {
			params := tinyQuantParams{scale: 0.015625, zero: 121}
			b.SetBytes(int64(tc.len * tc.conv.inChannels))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runTinyConvSIMD(tc.conv, input, params, output, tc.len)
			}
		})
		b.Run(tc.name+"/scalar", func(b *testing.B) {
			params := tinyQuantParams{scale: 0.015625, zero: 121}
			b.SetBytes(int64(tc.len * tc.conv.inChannels))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runTinyConvScalar(tc.conv, input, params, output, tc.len)
			}
		})
	}
}
