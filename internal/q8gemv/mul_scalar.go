// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || (!arm64 && (!amd64 || !amd64.v3))

package q8gemv

func mulKernel(x []float32, q []int8, scales, dst []float32, k, n int) {
	scalarMulInto(x, q, scales, dst, k, n)
}
