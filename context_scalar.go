// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || (!amd64 && !arm64)

package gophonic

func attentionContext4(scores0, scores1, scores2, scores3, values, output0, output1, output2, output3 []float32, headOffset int) {
	clear(output0[:headSize])
	clear(output1[:headSize])
	clear(output2[:headSize])
	clear(output3[:headSize])
	for key := 0; key < sequenceLength; key++ {
		value := values[key*hiddenSize+headOffset : key*hiddenSize+headOffset+headSize]
		w0, w1, w2, w3 := scores0[key], scores1[key], scores2[key], scores3[key]
		for d, x := range value {
			output0[d] += w0 * x
			output1[d] += w1 * x
			output2[d] += w2 * x
			output3[d] += w3 * x
		}
	}
}
