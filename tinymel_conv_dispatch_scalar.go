// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package gophonic

func runTinyConv(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength int) int {
	return runTinyConvScalar(conv, input, params, output, inputLength)
}

func runTinyConvRange(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength, fromT, toT int) int {
	return runTinyConvScalarRange(conv, input, params, output, inputLength, fromT, toT)
}
