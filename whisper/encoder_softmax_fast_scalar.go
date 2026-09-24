// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package whisper

// softmaxExpRowFallback replaces x with exp(x-max(x)) and returns 1/sum.
func softmaxExpRowFallback(x []float32) float32 {
	maxValue := x[0]
	for _, value := range x[1:] {
		maxValue = max(maxValue, value)
	}
	return 1 / softmaxExpInPlace(x, maxValue)
}
