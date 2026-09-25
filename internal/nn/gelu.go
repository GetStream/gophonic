// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package nn

import "math"

// GELU is x*Phi(x). A short Taylor expansion of the normal survival function
// around the nearest 1/128 grid point avoids per-element erf/exp calls while
// evaluating the same erf-based activation as PyTorch. The shared read-only
// table holds Q(r) and the normal density phi(r), both rounded from float64.
// For d=x-r:
//
//	Q(r+d) = Q(r) - phi(r)*d*(1-r*d/2+(r*r-1)*d*d/6) + O(d^4).
//
// With |d|<=1/256 the omitted term is below 1e-11; FP32 arithmetic dominates
// the error. Using Q directly also avoids cancellation on the negative tail.
// The scalar and SIMD kernels are tested against the double-precision erf
// definition with error <= 2e-7 + 1e-7*abs(GELU(x)), including cell boundaries.
var geluNormalTable = func() [1025][2]float32 {
	var table [1025][2]float32
	for i := range table {
		r := float64(i) / 128
		table[i][0] = float32(0.5 * math.Erfc(r/math.Sqrt2))
		table[i][1] = float32(math.Exp(-r*r/2) / math.Sqrt(2*math.Pi))
	}
	return table
}()

func geluScalar(values []float32) {
	for i, x := range values {
		a := float32(math.Abs(float64(x)))
		if !(a < 8) {
			// This includes NaN and infinities and preserves their IEEE behavior.
			values[i] = float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
			continue
		}
		index := int(a*128 + 0.5)
		r := float32(index) * (1.0 / 128)
		d := a - r
		entry := geluNormalTable[index]
		q := entry[0] - entry[1]*d*(1-d*(r*0.5-d*((r*r-1)*(1.0/6))))
		if x > 0 {
			q = 1 - q
		}
		values[i] = x * q
	}
}
