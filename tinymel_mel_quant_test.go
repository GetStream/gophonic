// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gofloor

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func TestTinyMelQuantizeSIMDExactParity(t *testing.T) {
	const n = tinyMelCount * tinyFrameCount
	rng := rand.New(rand.NewSource(0x4d454c))
	cases := make([][]float32, 0, 8)
	zero := make([]float32, n)
	cases = append(cases, zero)
	tone := make([]float32, n)
	for i := range tone {
		tone[i] = float32(math.Sin(float64(i)*0.019)*7 + math.Cos(float64(i)*0.0031)*2)
	}
	cases = append(cases, tone)
	pattern := make([]float32, n)
	values := [...]float32{-17.25, -1.5, -0.5, 0, 0.5, 1.5, 23.75}
	for i := range pattern {
		pattern[i] = values[i%len(values)]
	}
	cases = append(cases, pattern)
	random := make([]float32, n)
	for i := range random {
		random[i] = float32(rng.NormFloat64() * 31)
	}
	cases = append(cases, random)
	for _, exceptional := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		input := append([]float32(nil), tone...)
		input[0], input[15], input[16], input[17], input[n-1] = exceptional, exceptional, exceptional, exceptional, exceptional
		cases = append(cases, input)
	}

	for ci, input := range cases {
		want := make([]uint8, n)
		got := make([]uint8, n)
		scratch := make([]uint8, n)
		wantParams := quantizeTinyMel(input, want)
		gotParams := quantizeTinyMelSIMD(input, got, scratch)
		if math.Float32bits(gotParams.scale) != math.Float32bits(wantParams.scale) || gotParams.zero != wantParams.zero || !bytes.Equal(got, want) {
			t.Fatalf("case=%d params=%+v want=%+v bytes equal=%v", ci, gotParams, wantParams, bytes.Equal(got, want))
		}
		for _, tile := range [][2]int{{8, 8}, {16, 8}, {16, 16}, {16, 32}, {32, 16}, {32, 32}, {64, 16}} {
			transposeTinyMelBytesTile(scratch, got, tile[0], tile[1])
			if !bytes.Equal(got, want) {
				t.Fatalf("case=%d transpose tile=%dx%d differs", ci, tile[0], tile[1])
			}
		}
		transposeTinyMelBytesNaive(scratch, got)
		if !bytes.Equal(got, want) {
			t.Fatalf("case=%d naive transpose differs", ci)
		}
		if allocs := testing.AllocsPerRun(5, func() { quantizeTinyMelSIMD(input, got, scratch) }); allocs != 0 {
			t.Fatalf("case=%d SIMD mel quantizer allocated %g objects", ci, allocs)
		}
	}
}

func BenchmarkTinyMelQuantizeLayout(b *testing.B) {
	const n = tinyMelCount * tinyFrameCount
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.019)*7 + math.Cos(float64(i)*0.0031)*2)
	}
	output := make([]uint8, n)
	scratch := make([]uint8, n)
	bench := []struct {
		name string
		run  func()
	}{
		{name: "scalar-direct", run: func() { quantizeTinyMel(input, output) }},
		{name: "simd-naive-transpose", run: func() {
			quantizeTinySIMD(input, scratch)
			transposeTinyMelBytesNaive(scratch, output)
		}},
		{name: "simd-tile8x8", run: func() {
			quantizeTinySIMD(input, scratch)
			transposeTinyMelBytesTile(scratch, output, 8, 8)
		}},
		{name: "simd-tile16x8", run: func() {
			quantizeTinySIMD(input, scratch)
			transposeTinyMelBytesTile(scratch, output, 16, 8)
		}},
		{name: "simd-tile16x16", run: func() { quantizeTinyMelSIMD(input, output, scratch) }},
		{name: "simd-tile16x32", run: func() {
			quantizeTinySIMD(input, scratch)
			transposeTinyMelBytesTile(scratch, output, 16, 32)
		}},
		{name: "simd-tile32x16", run: func() {
			quantizeTinySIMD(input, scratch)
			transposeTinyMelBytesTile(scratch, output, 32, 16)
		}},
		{name: "simd-tile32x32", run: func() {
			quantizeTinySIMD(input, scratch)
			transposeTinyMelBytesTile(scratch, output, 32, 32)
		}},
		{name: "simd-tile64x16", run: func() {
			quantizeTinySIMD(input, scratch)
			transposeTinyMelBytesTile(scratch, output, 64, 16)
		}},
	}
	for _, tc := range bench {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				tc.run()
			}
		})
	}
}
