// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package gofloor

func tinyGRUProjectInputs(input, weights, projected []float32, sequenceLength, inputSize, outputSize int) {
	tinyGRUProjectInputsRowwise(input, weights, projected, sequenceLength, inputSize, outputSize)
}
