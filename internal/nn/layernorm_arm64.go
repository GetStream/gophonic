// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package nn

// Accelerated reports whether LayerNorm and ResidualNorm run NEON kernels
// for rows whose length is a positive multiple of eight.
const Accelerated = true

// layerNormNEON requires n to be a positive multiple of eight and every
// pointer to address at least n values.
//
//go:noescape
func layerNormNEON(src, dst, gamma, beta *float32, n int)

// residualNormNEON adds add (+ bias when non-nil) to row in place, then writes
// LayerNorm(row) to dst. The sum row + (add + bias) keeps the scalar order.
//
//go:noescape
func residualNormNEON(row, dst, gamma, beta *float32, n int, add, bias *float32)
