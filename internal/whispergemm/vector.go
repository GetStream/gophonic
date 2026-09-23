// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

// MulVector writes dst[rows] = weights[rows,len(x)] * x without packing.
// weightStride is measured in elements; output rows and the reduction length
// may be zero. Input/output overlap is unsupported. The operation allocates
// no memory and can run concurrently over disjoint output rows. SIMD uses
// fused FP32 multiply-add and a different reduction order from scalar builds.
func MulVector(dst, weights []float32, weightStride int, x []float32, rows int) error {
	if rows < 0 || len(dst) < rows || !validMatrix(weights, rows, len(x), weightStride) {
		return ErrShape
	}
	if len(x) == 0 {
		clear(dst[:rows])
		return nil
	}
	mulVector(dst, weights, weightStride, x, rows)
	return nil
}

func mulVectorScalar(dst, weights []float32, stride int, x []float32, rows int) {
	r := 0
	for ; r+4 <= rows; r += 4 {
		w0 := weights[r*stride : r*stride+len(x)]
		w1 := weights[(r+1)*stride : (r+1)*stride+len(x)]
		w2 := weights[(r+2)*stride : (r+2)*stride+len(x)]
		w3 := weights[(r+3)*stride : (r+3)*stride+len(x)]
		var c00, c01, c10, c11, c20, c21, c30, c31 float32
		p := 0
		for ; p+2 <= len(x); p += 2 {
			x0, x1 := x[p], x[p+1]
			c00 += float32(x0 * w0[p])
			c01 += float32(x1 * w0[p+1])
			c10 += float32(x0 * w1[p])
			c11 += float32(x1 * w1[p+1])
			c20 += float32(x0 * w2[p])
			c21 += float32(x1 * w2[p+1])
			c30 += float32(x0 * w3[p])
			c31 += float32(x1 * w3[p+1])
		}
		s0, s1, s2, s3 := c00+c01, c10+c11, c20+c21, c30+c31
		if p < len(x) {
			v := x[p]
			s0 += float32(v * w0[p])
			s1 += float32(v * w1[p])
			s2 += float32(v * w2[p])
			s3 += float32(v * w3[p])
		}
		dst[r], dst[r+1], dst[r+2], dst[r+3] = s0, s1, s2, s3
	}
	for ; r < rows; r++ {
		w := weights[r*stride : r*stride+len(x)]
		var sum float32
		for p, v := range x {
			sum += float32(v * w[p])
		}
		dst[r] = sum
	}
}
