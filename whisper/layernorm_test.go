// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"testing"
)

func TestArgmaxFiniteMatchesScalar(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	cases := [][]float32{
		{1, 2, 3}, {nan, nan}, {float32(math.Inf(-1)), nan},
	}
	long := make([]float32, 51864)
	for i := range long {
		long[i] = float32(math.Sin(float64(i) * 0.37))
	}
	long[40000], long[40001] = 7, 7 // ties keep the first
	cases = append(cases, long)
	withNaN := append([]float32(nil), long...)
	withNaN[3] = nan
	cases = append(cases, withNaN)
	withInf := append([]float32(nil), long...)
	withInf[51860] = inf
	cases = append(cases, withInf)
	allNaN := make([]float32, 64)
	for i := range allNaN {
		allNaN[i] = nan
	}
	cases = append(cases, allNaN)
	for i, values := range cases {
		want := -1
		best := float32(math.Inf(-1))
		for id, v := range values {
			if v > best {
				want, best = id, v
			}
		}
		if got := argmaxFinite(values); got != want {
			t.Fatalf("case %d: got %d want %d", i, got, want)
		}
	}
}

func TestResidualNormMatchesSeparatePasses(t *testing.T) {
	const rows, width = 9, 384
	for _, withBias := range []bool{false, true} {
		dst, add := make([]float32, rows*width), make([]float32, rows*width)
		gamma, beta, bias := make([]float32, width), make([]float32, width), make([]float32, width)
		for i := range dst {
			dst[i] = float32(math.Sin(float64(i)*0.11)) * 4
			add[i] = float32(math.Cos(float64(i)*0.07)) * 2
		}
		for i := range gamma {
			gamma[i], beta[i], bias[i] = float32(1+i%5)*0.3, float32(i%7)*0.1, float32(i%3)*0.2
		}
		wantDst, wantOut := append([]float32(nil), dst...), make([]float32, len(dst))
		var b []float32
		if withBias {
			b = bias
		}
		for r := 0; r < rows; r++ {
			row, a := wantDst[r*width:(r+1)*width], add[r*width:(r+1)*width]
			for i := range row {
				if b != nil {
					row[i] += a[i] + b[i]
				} else {
					row[i] += a[i]
				}
			}
			layerNormReference(row, wantOut[r*width:(r+1)*width], gamma, beta)
		}
		out := append([]float32(nil), add...) // out aliases add, as in the encoder
		op := encoderRows{kind: rowsResidual, dst: dst, src: out, bias: b, out: out, normW: gamma, normB: beta, width: width}
		op.ApplyRows(0, rows)
		for i := range dst {
			if dst[i] != wantDst[i] {
				t.Fatalf("bias=%v residual %d: %v != %v", withBias, i, dst[i], wantDst[i])
			}
			if d := math.Abs(float64(out[i] - wantOut[i])); d > 1e-6*(1+math.Abs(float64(wantOut[i]))) {
				t.Fatalf("bias=%v norm %d: %v != %v", withBias, i, out[i], wantOut[i])
			}
		}
	}
}

// layerNormReference is LayerNorm with float64 statistics, epsilon 1e-5, and
// one sequential accumulator.
func layerNormReference(src, dst, gamma, beta []float32) {
	var sum float64
	for _, x := range src {
		sum += float64(x)
	}
	mean := sum / float64(len(src))
	var variance float64
	for _, x := range src {
		delta := float64(x) - mean
		variance += delta * delta
	}
	variance /= float64(len(src))
	invStd := 1 / math.Sqrt(variance+1e-5)
	for i, x := range src {
		dst[i] = float32((float64(x)-mean)*invStd*float64(gamma[i]) + float64(beta[i]))
	}
}
