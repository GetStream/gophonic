// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gophonic

func runTinyConv(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength int) int {
	pad := conv.kernel / 2
	outputLength := (inputLength+2*pad-conv.kernel)/conv.stride + 1
	return runTinyConvRange(conv, input, params, output, inputLength, 0, outputLength)
}

func runTinyConvRange(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength, fromT, toT int) int {
	if conv.groups == conv.inChannels {
		return runTinyDepthwiseSIMDRange(conv, conv.depthwisePacked, input, params, output, inputLength, fromT, toT)
	}
	if conv.inChannels == tinyMelCount && conv.outChannels == tinyStemChannels && conv.groups == 1 && conv.kernel == 5 && conv.stride == 2 {
		return runTinyStemTile2x4Range(conv, input, params, output, inputLength, fromT, toT)
	}
	return runTinyConvTile4Range(conv, input, params, output, inputLength, fromT, toT)
}
