// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"math"
	"testing"
)

func TestMulVectorOracleAndTails(t *testing.T) {
	for _, rows := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9} {
		for _, columns := range []int{0, 1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 64, 65, 384, 1536} {
			stride := columns + 3
			// Both input slices have a one-float offset, deliberately removing
			// vector alignment. Sentinel output elements bound all writes.
			input := make([]float32, columns+1)[1:]
			weights := make([]float32, rows*stride+1)[1:]
			storage := make([]float32, rows+2)
			storage[0], storage[len(storage)-1] = -37, -37
			output := storage[1 : rows+1]
			state := uint32(0x12345678)
			for i := range input {
				input[i] = nextValue(&state)
			}
			for i := range weights {
				weights[i] = nextValue(&state)
			}
			if err := MulVector(output, weights, stride, input, rows); err != nil {
				t.Fatal(err)
			}
			for row, got := range output {
				want, sumAbs := reference64(input, columns, weights, stride, 0, row, columns)
				limit := 4 * float64(columns+1) * (1.0 / (1 << 24)) * sumAbs
				if math.IsNaN(float64(got)) || math.Abs(float64(got)-want) > limit {
					t.Fatalf("rows=%d columns=%d output[%d]=%g want %g (limit %g)", rows, columns, row, got, want, limit)
				}
			}
			if storage[0] != -37 || storage[len(storage)-1] != -37 {
				t.Fatal("output sentinel overwritten")
			}
		}
	}
}

func TestMulVectorSpecialValidationAndAllocations(t *testing.T) {
	for _, value := range []float32{0, float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		input, weights, output := make([]float32, 17), make([]float32, 5*17), make([]float32, 5)
		for i := range input {
			input[i] = 1
		}
		for i := range weights {
			weights[i] = value
		}
		if err := MulVector(output, weights, 17, input, 5); err != nil {
			t.Fatal(err)
		}
		want := value * 17
		for _, got := range output {
			if got != want && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
				t.Fatalf("special value %g: got %g want %g", value, got, want)
			}
		}
	}
	input, weights, output := make([]float32, 384), make([]float32, 7*384), make([]float32, 7)
	if MulVector(output, weights, 383, input, 7) != ErrShape || MulVector(output[:6], weights, 384, input, 7) != ErrShape || MulVector(output, weights[:len(weights)-1], 384, input, 7) != ErrShape || MulVector(output, weights, 384, input, -1) != ErrShape {
		t.Fatal("invalid shape accepted")
	}
	if allocations := testing.AllocsPerRun(10, func() {
		if err := MulVector(output, weights, 384, input, 7); err != nil {
			panic(err)
		}
	}); allocations != 0 {
		t.Fatalf("MulVector allocated %g objects", allocations)
	}
}

func BenchmarkMulVector(b *testing.B) {
	for _, s := range benchmarkShapes {
		if s.m != 1 {
			continue
		}
		b.Run(s.name, func(b *testing.B) {
			a, weights, dst, _ := benchmarkData(b, 1, s.k, s.n)
			b.ReportAllocs()
			for b.Loop() {
				if err := MulVector(dst, weights, s.k, a, s.n); err != nil {
					b.Fatal(err)
				}
			}
			benchmarkSink = dst[len(dst)-1]
		})
	}
}
