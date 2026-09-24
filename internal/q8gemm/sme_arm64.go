// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64

package q8gemm

import "sync/atomic"

// smeMulF16 accumulates panels 64-column panels of packed int8 weights against
// up to 16 packed FP16 activation rows in FP32, multiplies each output by its
// column scale and row scale, and stores cols valid columns per row at
// strideBytes intervals.
//
//go:noescape
func smeMulF16(q *int8, kPairs, panels int, activation *uint16, dst *float32, cols, rows, strideBytes int, colScales, rowScales *float32) (retries int)

// smeMulF16W is smeMulF16 for FP16 weights in the same panel layout.
//
//go:noescape
func smeMulF16W(w *uint16, kPairs, panels int, activation *uint16, dst *float32, cols, rows, strideBytes int, colScales, rowScales *float32) (retries int)

// smeMulI8 multiplies int8 activations by int8 weights in the four-way
// [K/4][64][4] layout with exact int32 accumulation, then applies column and
// row scales.
//
//go:noescape
func smeMulI8(w *int8, kQuads, panels int, activation *int8, dst *float32, cols, rows, strideBytes int, colScales, rowScales *float32) (retries int)

//go:noescape
func smeVectorBytes() int

var (
	smeEnabled = smeSupported() && smeVectorBytes() == 64
	smeRetries atomic.Uint64
)
