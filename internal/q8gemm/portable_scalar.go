// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || (!arm64 && (!amd64 || !amd64.v3))

package q8gemm

// fmaPanel computes acc += a*w over one 64-column panel.
func fmaPanel(acc, w *[OutputPanel]float32, a float32) {
	for c := range acc {
		acc[c] += a * w[c]
	}
}
