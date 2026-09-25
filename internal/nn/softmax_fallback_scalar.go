// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package nn

import "math"

// softmaxExpFallback replaces x with exp(x-max(x)) and returns 1/sum.
func softmaxExpFallback(x []float32) float32 {
	maxValue := x[0]
	for _, value := range x[1:] {
		maxValue = max(maxValue, value)
	}
	var total float32
	for i, value := range x {
		p := float32(math.Exp(float64(value - maxValue)))
		x[i] = p
		total += p
	}
	return 1 / total
}
