// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gophonic

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func TestTinyQuantizeSIMDExactParity(t *testing.T) {
	cases := [][]float32{
		nil,
		make([]float32, 33),
		{-128, 127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 125.5, 126.5, 127},
		{0, 255, 0.5, 1.5, 2.5, 3.5, 253.5, 254.5, 255},
		{0, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32},
		{math.MaxFloat32, -math.MaxFloat32, 0},
	}
	// Exercise the actual sixteen-element vector quantizer for every edge
	// case, preserving the same range and also leaving a scalar tail.
	for _, values := range cases {
		if len(values) == 0 {
			continue
		}
		expanded := make([]float32, 4*len(values)+17)
		for i := range expanded {
			expanded[i] = values[i%len(values)]
		}
		cases = append(cases, expanded)
	}
	for _, value := range []float32{1, -1, 127.5, -127.5, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		for _, length := range []int{1, 15, 16, 17, 128} {
			values := make([]float32, length)
			for i := range values {
				values[i] = value
			}
			cases = append(cases, values)
		}
	}
	rng := rand.New(rand.NewSource(1967))
	for _, length := range []int{1, 7, 8, 15, 16, 17, 31, 32, 33, 256, 76800} {
		values := make([]float32, length)
		for i := range values {
			values[i] = float32(rng.NormFloat64()) * 30
		}
		cases = append(cases, values)
	}
	// Put each exceptional value both inside vector blocks and scalar tails.
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		for position := 0; position < 33; position++ {
			values := make([]float32, 33)
			for i := range values {
				values[i] = float32(i) - 16
			}
			values[position] = value
			cases = append(cases, values)
		}
	}
	for ci, input := range cases {
		got, want := make([]uint8, len(input)), make([]uint8, len(input))
		gotParams := quantizeTinySIMD(input, got)
		wantParams := quantizeTinyScalar(input, want)
		if math.Float32bits(gotParams.scale) != math.Float32bits(wantParams.scale) || gotParams.zero != wantParams.zero || !bytes.Equal(got, want) {
			t.Fatalf("case=%d length=%d params=%+v want=%+v bytesEqual=%v", ci, len(input), gotParams, wantParams, bytes.Equal(got, want))
		}
		if allocs := testing.AllocsPerRun(3, func() { quantizeTinySIMD(input, got) }); allocs != 0 {
			t.Fatalf("quantize allocated %g objects", allocs)
		}
	}
}

func BenchmarkTinyQuantizeSIMD(b *testing.B) {
	input := make([]float32, 400*192)
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.037))*17 + float32(math.Cos(float64(i)*0.011))
	}
	output := make([]uint8, len(input))
	for _, kernel := range []struct {
		name string
		run  func([]float32, []uint8) tinyQuantParams
	}{
		{"baseline", quantizeTinyScalar},
		{"simd", quantizeTinySIMD},
	} {
		b.Run(kernel.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				kernel.run(input, output)
			}
		})
	}
}
