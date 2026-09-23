// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gofloor

func quantizeTinyMelForInference(mel []float32, output, scratch []uint8) tinyQuantParams {
	return quantizeTinyMelSIMD(mel, output, scratch)
}
