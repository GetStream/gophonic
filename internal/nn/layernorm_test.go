// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package nn

import (
	"math"
	"testing"
)

func TestLayerNormMatchesGeneric(t *testing.T) {
	for _, n := range []int{8, 16, 384, 512, 768, 1024} {
		src, gamma, beta := make([]float32, n), make([]float32, n), make([]float32, n)
		for i := range src {
			src[i] = float32(math.Sin(float64(i)*0.7))*3 + 50 // offset stresses cancellation
			gamma[i] = float32(math.Cos(float64(i) * 0.3))
			beta[i] = float32(i%13) * 0.01
		}
		want, got := make([]float32, n), make([]float32, n+1)
		got[n] = -317
		layerNormGeneric(src, want, gamma, beta)
		LayerNorm(src, got, gamma, beta)
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > 1e-6*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("n=%d index %d: %v != %v", n, i, got[i], want[i])
			}
		}
		if got[n] != -317 {
			t.Fatalf("n=%d wrote past the row", n)
		}
	}
}

func TestResidualNormMatchesSeparatePasses(t *testing.T) {
	for _, width := range []int{384, 1024, 12} {
		for _, withBias := range []bool{false, true} {
			row, add := make([]float32, width), make([]float32, width)
			gamma, beta, bias := make([]float32, width), make([]float32, width), make([]float32, width)
			for i := range row {
				row[i] = float32(math.Sin(float64(i)*0.11)) * 4
				add[i] = float32(math.Cos(float64(i)*0.07)) * 2
				gamma[i], beta[i], bias[i] = float32(1+i%5)*0.3, float32(i%7)*0.1, float32(i%3)*0.2
			}
			var b []float32
			if withBias {
				b = bias
			}
			wantRow, wantOut := append([]float32(nil), row...), make([]float32, width)
			for i := range wantRow {
				if b != nil {
					wantRow[i] += add[i] + b[i]
				} else {
					wantRow[i] += add[i]
				}
			}
			layerNormGeneric(wantRow, wantOut, gamma, beta)
			out := add // dst aliases add, as encoders use it
			ResidualNorm(row, out, gamma, beta, add, b)
			for i := range row {
				if row[i] != wantRow[i] {
					t.Fatalf("width=%d bias=%v residual %d: %v != %v", width, withBias, i, row[i], wantRow[i])
				}
				if d := math.Abs(float64(out[i] - wantOut[i])); d > 1e-6*(1+math.Abs(float64(wantOut[i]))) {
					t.Fatalf("width=%d bias=%v norm %d: %v != %v", width, withBias, i, out[i], wantOut[i])
				}
			}
		}
	}
}

func TestSoftmaxExpMatchesReference(t *testing.T) {
	for _, n := range []int{1, 3, 4, 8, 12, 64, 104, 1500, 1501} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*1.3))*9 - float32(i%5)*20
		}
		x0 := append([]float32(nil), x...)
		ref := make([]float64, n)
		maxValue := math.Inf(-1)
		for _, v := range x {
			maxValue = math.Max(maxValue, float64(v))
		}
		var sum float64
		for i, v := range x {
			ref[i] = math.Exp(float64(v) - maxValue)
			sum += ref[i]
		}
		inverse := SoftmaxExp(x)
		for i := range x {
			got := float64(x[i]) * float64(inverse)
			want := ref[i] / sum
			// Rounding x-max to FP32 alone contributes |x-max|*2^-24 relative error.
			// A sequential FP32 sum adds up to n*2^-24 relative error.
			limit := (4e-7+math.Abs(float64(x0[i])-maxValue)*1.2e-7+float64(n)*6e-8)*want + 1e-38
			if math.Abs(got-want) > limit {
				t.Fatalf("n=%d index %d: %g != %g", n, i, got, want)
			}
		}
	}
}
