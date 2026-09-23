// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

// Scalar tiles reuse each loaded value across four rows and four columns.
// Explicit conversions pin a separately rounded multiply and add in ascending
// K order; Go must not fuse across an explicit float32 conversion.
func mulPackedScalar(dst []float32, dstStride int, a []float32, aStride int, packed []float32, m, k, n int) {
	for c := 0; c < n; c += 4 {
		width := min(4, n-c)
		panel := (c / panelColumns) * k * panelColumns
		weights := packed[panel+c%panelColumns:]
		r := 0
		if width == 4 {
			for ; r+4 <= m; r += 4 {
				scalar4x4(
					a[r*aStride:r*aStride+k], a[(r+1)*aStride:(r+1)*aStride+k],
					a[(r+2)*aStride:(r+2)*aStride+k], a[(r+3)*aStride:(r+3)*aStride+k], weights,
					dst[r*dstStride+c:r*dstStride+c+4], dst[(r+1)*dstStride+c:(r+1)*dstStride+c+4],
					dst[(r+2)*dstStride+c:(r+2)*dstStride+c+4], dst[(r+3)*dstStride+c:(r+3)*dstStride+c+4],
				)
			}
		}
		for ; r < m; r++ {
			scalar1x4(a[r*aStride:r*aStride+k], weights, dst[r*dstStride+c:r*dstStride+c+width])
		}
	}
}

func scalar4x4(a0, a1, a2, a3, weights, d0, d1, d2, d3 []float32) {
	k := len(a0)
	_ = a1[k-1]
	_ = a2[k-1]
	_ = a3[k-1]
	_ = weights[(k-1)*panelColumns+3]
	var c00, c01, c02, c03 float32
	var c10, c11, c12, c13 float32
	var c20, c21, c22, c23 float32
	var c30, c31, c32, c33 float32
	for p, x0 := range a0 {
		w := (*[4]float32)(weights[p*panelColumns:])
		w0, w1, w2, w3 := w[0], w[1], w[2], w[3]
		c00 += float32(x0 * w0)
		c01 += float32(x0 * w1)
		c02 += float32(x0 * w2)
		c03 += float32(x0 * w3)
		x1 := a1[p]
		c10 += float32(x1 * w0)
		c11 += float32(x1 * w1)
		c12 += float32(x1 * w2)
		c13 += float32(x1 * w3)
		x2 := a2[p]
		c20 += float32(x2 * w0)
		c21 += float32(x2 * w1)
		c22 += float32(x2 * w2)
		c23 += float32(x2 * w3)
		x3 := a3[p]
		c30 += float32(x3 * w0)
		c31 += float32(x3 * w1)
		c32 += float32(x3 * w2)
		c33 += float32(x3 * w3)
	}
	_ = d0[3]
	d0[0], d0[1], d0[2], d0[3] = c00, c01, c02, c03
	_ = d1[3]
	d1[0], d1[1], d1[2], d1[3] = c10, c11, c12, c13
	_ = d2[3]
	d2[0], d2[1], d2[2], d2[3] = c20, c21, c22, c23
	_ = d3[3]
	d3[0], d3[1], d3[2], d3[3] = c30, c31, c32, c33
}

func scalar1x4(a, weights, dst []float32) {
	var c0, c1, c2, c3 float32
	for p, x := range a {
		w := (*[4]float32)(weights[p*panelColumns:])
		c0 += float32(x * w[0])
		c1 += float32(x * w[1])
		c2 += float32(x * w[2])
		c3 += float32(x * w[3])
	}
	if len(dst) == 4 {
		dst[0], dst[1], dst[2], dst[3] = c0, c1, c2, c3
		return
	}
	tail := [4]float32{c0, c1, c2, c3}
	copy(dst, tail[:])
}
