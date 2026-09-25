// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package whisper

func attnPrepNEON(q, k, v, qbias, vbias *float32, n int, scale float32) { panic("unreachable") }

func maxNumNEON(x *float32, n int) float32 { panic("unreachable") }
