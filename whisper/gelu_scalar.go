// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || (!arm64 && (!amd64 || !amd64.v3))

package whisper

func applyGELU(values []float32) {
	applyGELUScalar(values)
}
