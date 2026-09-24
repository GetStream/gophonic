// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemv

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestMulIntoAgainstScalar(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 4096, 14336}
	rng := rand.New(rand.NewSource(0x5147454d56))
	for _, k := range lengths {
		for _, n := range []int{1, 3, 5} {
			t.Run(testShape(k, n), func(t *testing.T) {
				x := make([]float32, k)
				q := make([]int8, n*k)
				scales := make([]float32, n)
				got := make([]float32, n)
				want := make([]float32, n)
				for i := range x {
					x[i] = (rng.Float32()*2 - 1) * 4
				}
				for i := range q {
					q[i] = int8(rng.Intn(256) - 128)
				}
				for i := range scales {
					scales[i] = (rng.Float32()*2 - 1) * 0.1
				}
				scalarMulInto(x, q, scales, want, k, n)
				MulInto(x, q, scales, got, k, n)
				for i := range got {
					if !closeF32(got[i], want[i], 5e-4, 2e-5) {
						t.Fatalf("output[%d] = %.9g, scalar = %.9g", i, got[i], want[i])
					}
				}
			})
		}
	}
}

func TestMulIntoSignedExtremesAndTail(t *testing.T) {
	const k, n = 35, 3
	x := make([]float32, k)
	q := make([]int8, k*n)
	for i := range x {
		x[i] = float32((i%9)-4) * 0.25
	}
	for i := range q {
		switch i % 4 {
		case 0:
			q[i] = -128
		case 1:
			q[i] = 127
		case 2:
			q[i] = -1
		default:
			q[i] = 1
		}
	}
	scales := []float32{1, -0.03125, 0.0001}
	got, want := make([]float32, n), make([]float32, n)
	scalarMulInto(x, q, scales, want, k, n)
	MulInto(x, q, scales, got, k, n)
	for i := range got {
		if !closeF32(got[i], want[i], 1e-5, 1e-6) {
			t.Fatalf("output[%d] = %.9g, scalar = %.9g", i, got[i], want[i])
		}
	}
}

func TestMulIntoNonFiniteValues(t *testing.T) {
	x := make([]float32, 20)
	q := make([]int8, 20)
	x[0] = float32(math.Inf(1))
	x[4] = float32(math.NaN())
	x[19] = float32(math.Copysign(0, -1))
	q[0], q[4], q[19] = 0, 3, -4
	scales := []float32{2}
	got, want := make([]float32, 1), make([]float32, 1)
	scalarMulInto(x, q, scales, want, len(x), 1)
	MulInto(x, q, scales, got, len(x), 1)
	if !math.IsNaN(float64(want[0])) || !math.IsNaN(float64(got[0])) {
		t.Fatalf("got %v, scalar %v; expected NaN propagation", got[0], want[0])
	}
}

func TestMulIntoShapeChecks(t *testing.T) {
	tests := []struct {
		name string
		x    []float32
		q    []int8
		s    []float32
		dst  []float32
		k, n int
	}{
		{"negative-k", nil, nil, nil, nil, -1, 0},
		{"negative-n", nil, nil, nil, nil, 0, -1},
		{"activation-length", []float32{1}, nil, nil, nil, 0, 0},
		{"scale-length", nil, nil, []float32{1}, nil, 0, 0},
		{"output-length", nil, nil, nil, []float32{1}, 0, 0},
		{"weights-length", []float32{1}, []int8{1}, []float32{1, 1}, []float32{0, 0}, 1, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			MulInto(tt.x, tt.q, tt.s, tt.dst, tt.k, tt.n)
		})
	}
}

func TestMulIntoZeroAllocs(t *testing.T) {
	const k, n = 4096, 32
	x := make([]float32, k)
	q := make([]int8, n*k)
	scales := make([]float32, n)
	dst := make([]float32, n)
	MulInto(x, q, scales, dst, k, n)
	if got := testing.AllocsPerRun(100, func() { MulInto(x, q, scales, dst, k, n) }); got != 0 {
		t.Fatalf("MulInto allocations/run = %g, want 0", got)
	}
}

func BenchmarkMulInto(b *testing.B) {
	for _, k := range []int{4096, 14336} {
		b.Run(testShape(k, 4096), func(b *testing.B) {
			rng := rand.New(rand.NewSource(int64(k)))
			x := make([]float32, k)
			q := make([]int8, k*4096)
			scales := make([]float32, 4096)
			dst := make([]float32, 4096)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}
			for i := range q {
				q[i] = int8(rng.Intn(256) - 128)
			}
			for i := range scales {
				scales[i] = 0.01
			}
			b.SetBytes(int64(len(q)))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				MulInto(x, q, scales, dst, k, len(dst))
			}
		})
	}
}

func closeF32(a, b, absTol, relTol float32) bool {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return math.IsNaN(float64(a)) && math.IsNaN(float64(b))
	}
	if math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0) {
		return a == b
	}
	d := float32(math.Abs(float64(a - b)))
	limit := absTol + relTol*float32(math.Abs(float64(b)))
	return d <= limit
}

func testShape(k, n int) string {
	return fmt.Sprintf("k%d-n%d", k, n)
}
