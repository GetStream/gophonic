// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || (!amd64 && !arm64)

package int8probe

func probeDot4x4(a0, a1, a2, a3, b0, b1, b2, b3 []float32, out *[16]float32) {
	var c00, c01, c02, c03 float32
	var c10, c11, c12, c13 float32
	var c20, c21, c22, c23 float32
	var c30, c31, c32, c33 float32
	for i := range a0 {
		x0, x1, x2, x3 := a0[i], a1[i], a2[i], a3[i]
		y0, y1, y2, y3 := b0[i], b1[i], b2[i], b3[i]
		c00 += x0 * y0
		c01 += x0 * y1
		c02 += x0 * y2
		c03 += x0 * y3
		c10 += x1 * y0
		c11 += x1 * y1
		c12 += x1 * y2
		c13 += x1 * y3
		c20 += x2 * y0
		c21 += x2 * y1
		c22 += x2 * y2
		c23 += x2 * y3
		c30 += x3 * y0
		c31 += x3 * y1
		c32 += x3 * y2
		c33 += x3 * y3
	}
	*out = [16]float32{c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33}
}
