// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"math"
	"testing"
)

var mathExp = math.Exp

func TestExpNonPositive32(t *testing.T) {
	worst := 0.0
	for i := 0; i <= 200000; i++ {
		x := -87 * float64(i) / 200000
		got := float64(expNonPositive32(float32(x)))
		want := mathExp(float64(float32(x)))
		if x < -80 {
			// Near float32 underflow only an absolute bound is meaningful.
			if abs64(got-want) > 1e-34 {
				t.Fatalf("exp(%g) = %g, want %g", x, got, want)
			}
			continue
		}
		if rel := abs64(got-want) / want; rel > worst {
			worst = rel
		}
	}
	if worst > 1e-6 {
		t.Fatalf("worst relative error %.3g", worst)
	}
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestVectorKernelsMatchScalar(t *testing.T) {
	x := make([]float32, 1037)
	up := make([]float32, len(x))
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.37)) * float32(i%40-20)
		up[i] = float32(math.Cos(float64(i) * 0.11))
	}
	got := append([]float32(nil), x...)
	swigluInto(got, up)
	for i := range x {
		want := float64(x[i]) / (1 + math.Exp(-float64(x[i]))) * float64(up[i])
		if d := math.Abs(float64(got[i]) - want); d > 2e-6*math.Max(1, math.Abs(want)) {
			t.Fatalf("swiglu[%d](%g) = %g, want %g", i, x[i], got[i], want)
		}
	}
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	if d := math.Abs(float64(sumSquares(x)) - ss); d > 1e-5*ss {
		t.Fatalf("sumSquares = %g, want %g", sumSquares(x), ss)
	}
	var dot float64
	for i := range x {
		dot += float64(x[i]) * float64(up[i])
	}
	if d := math.Abs(float64(dot32(x, up)) - dot); d > 1e-4 {
		t.Fatalf("dot32 = %g, want %g", dot32(x, up), dot)
	}
}
