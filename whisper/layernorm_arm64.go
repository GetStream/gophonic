// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

const layerNormAccelerated = true

// layerNormNEON requires n to be a positive multiple of eight and every
// pointer to address at least n values.
//
//go:noescape
func layerNormNEON(src, dst, gamma, beta *float32, n int)

// attnPrepNEON applies q = (q+qbias)*scale, k *= scale, and v += vbias to n
// values of one row each; n must be a positive multiple of four.
//
//go:noescape
func attnPrepNEON(q, k, v, qbias, vbias *float32, n int, scale float32)
