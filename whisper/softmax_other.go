// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package whisper

const softmaxAccelerated = false

var softmaxConstants [11]float32

func softmaxExpNEON(x *float32, n int, constants *[11]float32) float32 { panic("unreachable") }
