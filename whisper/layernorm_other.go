// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package whisper

const layerNormAccelerated = false

func layerNormNEON(src, dst, gamma, beta *float32, n int) { panic("unreachable") }
