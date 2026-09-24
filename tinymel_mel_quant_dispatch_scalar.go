// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package gophonic

func quantizeTinyMelForInference(mel []float32, output, scratch []uint8) tinyQuantParams {
	return quantizeTinyMel(mel, output)
}
