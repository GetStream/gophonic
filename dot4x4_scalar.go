// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || (!amd64 && !arm64)

package gophonic

func dotProduct4x4(a0, a1, a2, a3, b0, b1, b2, b3 []float32, out *[16]float32) {
	var s00, s01, s02, s03 float32
	var s10, s11, s12, s13 float32
	var s20, s21, s22, s23 float32
	var s30, s31, s32, s33 float32
	for i := range a0 {
		x0, x1, x2, x3 := a0[i], a1[i], a2[i], a3[i]
		y0, y1, y2, y3 := b0[i], b1[i], b2[i], b3[i]
		s00 += x0 * y0
		s01 += x0 * y1
		s02 += x0 * y2
		s03 += x0 * y3
		s10 += x1 * y0
		s11 += x1 * y1
		s12 += x1 * y2
		s13 += x1 * y3
		s20 += x2 * y0
		s21 += x2 * y1
		s22 += x2 * y2
		s23 += x2 * y3
		s30 += x3 * y0
		s31 += x3 * y1
		s32 += x3 * y2
		s33 += x3 * y3
	}
	*out = [16]float32{s00, s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23, s30, s31, s32, s33}
}
