// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package nn

// Accelerated reports whether LayerNorm and ResidualNorm run NEON kernels
// for rows whose length is a positive multiple of eight.
const Accelerated = false

func layerNormNEON(src, dst, gamma, beta *float32, n int) { panic("unreachable") }

func residualNormNEON(row, dst, gamma, beta *float32, n int, add, bias *float32) {
	panic("unreachable")
}
