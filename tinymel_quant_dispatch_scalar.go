// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package gofloor

func quantizeTiny(input []float32, output []uint8) tinyQuantParams {
	return quantizeTinyScalar(input, output)
}
