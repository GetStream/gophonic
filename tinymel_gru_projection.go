// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

// tinyGRUProjectInputsRowwise retains the existing three-gate GEMV traversal.
// weights is [three*hidden,inputSize], projected is [time,three*hidden].
func tinyGRUProjectInputsRowwise(input, weights, projected []float32, sequenceLength, inputSize, outputSize int) {
	hiddenSize := outputSize / tinyGRUGateCount
	for t := 0; t < sequenceLength; t++ {
		x := input[t*inputSize : (t+1)*inputSize]
		y := projected[t*outputSize : (t+1)*outputSize]
		for gate := 0; gate < tinyGRUGateCount; gate++ {
			w := weights[gate*hiddenSize*inputSize : (gate+1)*hiddenSize*inputSize]
			tinyGRUMatVec(x, w, y[gate*hiddenSize:(gate+1)*hiddenSize], inputSize, hiddenSize)
		}
	}
}
