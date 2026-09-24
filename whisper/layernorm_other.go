// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package whisper

const layerNormAccelerated = false

func layerNormNEON(src, dst, gamma, beta *float32, n int) { panic("unreachable") }

func attnPrepNEON(q, k, v, qbias, vbias *float32, n int, scale float32) { panic("unreachable") }

func maxNumNEON(x *float32, n int) float32 { panic("unreachable") }

func residualNormNEON(row, dst, gamma, beta *float32, n int, add, bias *float32) { panic("unreachable") }
