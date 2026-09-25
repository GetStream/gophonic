// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func TestGELUAccuracy(t *testing.T) {
	inputs := make([]float32, 0, 150000)
	for i := range 65537 {
		inputs = append(inputs, float32(i)*(20.0/65536)-10)
	}
	// Grid rounding boundaries and their adjacent FP32 values exercise both
	// sides of every interpolation cell, including the vector lookup limit.
	for i := range 1025 {
		x := (float32(i) + 0.5) / 128
		for _, v := range []float32{math.Nextafter32(x, 0), x, math.Nextafter32(x, float32(math.Inf(1)))} {
			inputs = append(inputs, v, -v)
		}
	}
	rng := rand.New(rand.NewPCG(19, 73))
	for range 65536 {
		inputs = append(inputs, math.Float32frombits(rng.Uint32()))
	}
	// Signaling NaNs can make an unchecked SIMD table index escape the table.
	for _, bits := range []uint32{0x7f800001, 0xff800001, 0x7f9fffff, 0xff9fffff} {
		inputs = append(inputs, math.Float32frombits(bits))
	}
	for _, kernel := range []struct {
		name string
		fn   func([]float32)
	}{{"dispatched", GELU}, {"scalar", geluScalar}, {"fused", func(v []float32) {
		n := len(v) / 4 * 4
		BiasGELU(v[:n], nil, 1, n)
		GELU(v[n:])
	}}} {
		t.Run(kernel.name, func(t *testing.T) {
			got := append([]float32(nil), inputs...)
			kernel.fn(got)
			var maxAbs, maxScaled float64
			for i, x := range inputs {
				want := float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
				if math.IsNaN(float64(want)) {
					if !math.IsNaN(float64(got[i])) {
						t.Fatalf("GELU(%g)=%g, want NaN", x, got[i])
					}
					continue
				}
				if want == got[i] {
					continue
				}
				diff := math.Abs(float64(got[i]) - float64(want))
				bound := 2e-7 + 1e-7*math.Abs(float64(want))
				maxAbs = max(maxAbs, diff)
				maxScaled = max(maxScaled, diff/bound)
				if math.IsNaN(float64(got[i])) || diff > bound {
					t.Fatalf("GELU(%g)=%.9g, want %.9g; error=%g limit=%g", x, got[i], want, diff, bound)
				}
			}
			t.Logf("maximum absolute error=%g, largest fraction of mixed error bound=%g", maxAbs, maxScaled)
		})
	}
}

func TestGELULengthsAlignmentSpecialAndAllocation(t *testing.T) {
	for _, size := range []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 384, 1536, 1537} {
		for offset := range 4 {
			t.Run(fmt.Sprintf("n%d_offset%d", size, offset), func(t *testing.T) {
				buf := make([]float32, size+offset+1)
				buf[len(buf)-1] = 19
				values := buf[offset : offset+size]
				for i := range values {
					values[i] = float32(i%41-20) / 4
				}
				GELU(values)
				for i, got := range values {
					x := float64(float32(i%41-20) / 4)
					want := 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
					if math.Abs(float64(got)-want) > 2e-7+1e-7*math.Abs(want) {
						t.Fatalf("GELU(%g)=%g, want %g", x, got, want)
					}
				}
				if buf[len(buf)-1] != 19 {
					t.Fatal("kernel overwrote suffix")
				}
			})
		}
	}
	values := []float32{0, math.Float32frombits(1 << 31), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), -9, 9, 1}
	GELU(values)
	if math.Float32bits(values[0]) != 0 || math.Float32bits(values[1]) != 1<<31 || !math.IsInf(float64(values[2]), 1) || !math.IsNaN(float64(values[3])) || !math.IsNaN(float64(values[4])) {
		t.Fatalf("IEEE special-value behavior changed: %v", values)
	}
	buf := make([]float32, 1536)
	if allocs := testing.AllocsPerRun(20, func() {
		for i := range buf {
			buf[i] = float32(i%41-20) / 4
		}
		GELU(buf)
	}); allocs != 0 {
		t.Fatalf("GELU allocated %g objects", allocs)
	}
}

func TestBiasGELUMatchesSeparatePasses(t *testing.T) {
	const rows, width = 5, 1536
	values, bias := make([]float32, rows*width), make([]float32, width)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.013)) * 12
	}
	for i := range bias {
		bias[i] = float32(i%9-4) * 0.75
	}
	want := append([]float32(nil), values...)
	AddRowBias(want, bias, rows, width)
	for i, x := range want {
		want[i] = float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
	}
	BiasGELU(values, bias, rows, width)
	for i := range values {
		if d := math.Abs(float64(values[i] - want[i])); d > 2e-7+1e-7*math.Abs(float64(want[i])) {
			t.Fatalf("index %d: %g != %g", i, values[i], want[i])
		}
	}
}

var geluBenchmarkSink float32

func BenchmarkGELU(b *testing.B) {
	for _, size := range []int{384, 1536, 384 * 32, 1536 * 32} {
		for _, kernel := range []struct {
			name string
			fn   func([]float32)
		}{
			{"dispatched", GELU},
			{"scalar", geluScalar},
			{"erf", func(values []float32) {
				for i, x := range values {
					values[i] = float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
				}
			}},
		} {
			b.Run(fmt.Sprintf("%s/n%d", kernel.name, size), func(b *testing.B) {
				original := make([]float32, size)
				for i := range original {
					original[i] = float32(i%401-200) * 0.025
				}
				values := make([]float32, size)
				b.ReportAllocs()
				for b.Loop() {
					// Refill is included equally so repeated GELU cannot collapse
					// the input into zeros and bypass representative arithmetic.
					copy(values, original)
					kernel.fn(values)
				}
				geluBenchmarkSink = values[len(values)-1]
			})
		}
	}
}
