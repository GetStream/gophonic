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

// smeRowI8 multiplies one contiguous int8 activation row by int8 weight
// panels with SME2 multi-vector SDOT; K must be a multiple of 32.
//
//go:noescape
func smeRowI8(w *int8, kGroups, panels int, activation *int8, dst *float32, cols int, colScales, rowScale *float32) (retries int)

// smeRowF16 multiplies one contiguous FP16 activation row by FP16 weight
// panels with SME2 multi-vector FDOT; K must be a multiple of 16.
//
//go:noescape
func smeRowF16(w *uint16, kGroups, panels int, activation *uint16, dst *float32, cols int, colScales, rowScale *float32) (retries int)

//go:noescape
func smeVectorBytes() int

var (
	smeEnabled = smeSupported() && smeVectorBytes() == 64
	smeRetries atomic.Uint64
)
