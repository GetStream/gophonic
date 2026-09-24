// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package whisper

import "math"

func softmaxExpInPlace(x []float32, maxValue float32) float32 {
	var total float32
	for i, value := range x {
		p := float32(math.Exp(float64(value - maxValue)))
		x[i] = p
		total += p
	}
	return total
}
