// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package nn

func GELU(values []float32) {
	geluScalar(values)
}
