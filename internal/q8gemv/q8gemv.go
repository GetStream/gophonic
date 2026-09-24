// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package q8gemv computes one float32 activation vector against rows of
// signed-int8 weights with one float32 scale per row.
//
// It is weight-only quantization: activations stay float32, and no scratch
// buffer or activation quantization is required. The SIMD implementation
// reassociates the float32 reduction, so bit-for-bit scalar parity is not
// promised; tests bound that rounding difference against the scalar oracle.
package q8gemv

// MulInto computes dst[j] = dot(x, float32(q[j*k:(j+1)*k])) * scales[j].
//
// The output and all inputs are caller-owned. MulInto does not allocate. The
// input slices must not overlap dst. It panics if dimensions or slice lengths
// do not match.
func MulInto(x []float32, q []int8, scales []float32, dst []float32, k, n int) {
	if k < 0 || n < 0 || len(x) != k || len(scales) != n || len(dst) != n {
		panic("q8gemv: invalid dimensions")
	}
	if k == 0 {
		if len(q) != 0 {
			panic("q8gemv: invalid weight length")
		}
	} else if len(q)%k != 0 || len(q)/k != n {
		panic("q8gemv: invalid weight length")
	}
	mulKernel(x, q, scales, dst, k, n)
}
