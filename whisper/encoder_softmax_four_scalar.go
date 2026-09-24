// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package whisper

func softmaxFourRows(x0, x1, x2, x3 []float32) {
	softmaxRow(x0)
	softmaxRow(x1)
	softmaxRow(x2)
	softmaxRow(x3)
}
