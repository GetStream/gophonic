// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package whispergemm

func mulVector(dst, weights []float32, stride int, x []float32, rows int) {
	mulVectorScalar(dst, weights, stride, x, rows)
}
