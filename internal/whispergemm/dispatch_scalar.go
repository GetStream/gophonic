// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package whispergemm

const kernelName = "scalar"

func mulPacked(dst []float32, dstStride int, a []float32, aStride int, packed []float32, m, k, n int) {
	mulPackedScalar(dst, dstStride, a, aStride, packed, m, k, n)
}
