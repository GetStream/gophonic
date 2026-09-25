// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !(goexperiment.simd && arm64)

package qwen3tts

// apply writes SnakeBeta of rows of n channels from src to dst.
func (s *snake) apply(dst, src []float32, n int) { snakeScalar(dst, src, s.a, s.invB, n) }
