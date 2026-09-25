// Copyright 2026 The gophonic authors
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

package nn

import "math"

const softmaxAccelerated = true

// softmaxConstants feeds softmaxExpNEON: the ln(2^-126) input floor, ggml's
// NEON exponential constants, and the FP32 bits of 1.0.
var softmaxConstants = [11]float32{
	-87.33654, 0x1.8p23, 0x1.715476p+0, -0x1.62e4p-1, -0x1.7f7d1cp-20,
	0x1.ffffecp-1, 0x1.fffdb6p-2, 0x1.555e66p-3, 0x1.573e2ep-5, 0x1.0e4020p-7,
	math.Float32frombits(0x3f800000),
}

// softmaxExpNEON replaces x[:n] with exp(x-max(x)) and returns their sum.
// n must be a positive multiple of four.
//
//go:noescape
func softmaxExpNEON(x *float32, n int, constants *[11]float32) float32
