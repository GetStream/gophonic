// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemv

// scalarMulInto is the portable definition and the fallback on builds without
// the ARM64 Go SIMD experiment.
func scalarMulInto(x []float32, q []int8, scales, dst []float32, k, n int) {
	for row := range n {
		weights := q[row*k : (row+1)*k]
		var sum float32
		for col, w := range weights {
			sum += x[col] * float32(w)
		}
		dst[row] = sum * scales[row]
	}
}
