// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

const layerNormAccelerated = true

// layerNormNEON requires n to be a positive multiple of eight and every
// pointer to address at least n values.
//
//go:noescape
func layerNormNEON(src, dst, gamma, beta *float32, n int)
