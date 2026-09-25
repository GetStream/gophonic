// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !(goexperiment.simd && arm64)

package qwen3tts

// apply writes SnakeBeta of rows of n channels from src, plus bias when
// given, to dst.
func (s *snake) apply(dst, src, bias []float32, n int) { snakeScalar(dst, src, bias, s.a, s.invB, n) }

func addTo(dst, src []float32) { addToScalar(dst, src) }

// residual writes rows of n channels of src + z + bias to dst.
func residual(dst, src, z, bias []float32, n int) { residualScalar(dst, src, z, bias, n) }
