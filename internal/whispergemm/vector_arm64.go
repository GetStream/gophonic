// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package whispergemm

import "simd/archsimd"

func mulVector(dst, weights []float32, stride int, x []float32, rows int) {
	r := 0
	for ; r+4 <= rows; r += 4 {
		vector4(dst[r:r+4], x,
			weights[r*stride:r*stride+len(x)], weights[(r+1)*stride:(r+1)*stride+len(x)],
			weights[(r+2)*stride:(r+2)*stride+len(x)], weights[(r+3)*stride:(r+3)*stride+len(x)])
	}
	for ; r < rows; r++ {
		dst[r] = vectorDot(x, weights[r*stride:r*stride+len(x)])
	}
}

func vector4(dst, x, w0, w1, w2, w3 []float32) {
	k := len(x)
	w0, w1, w2, w3 = w0[:k], w1[:k], w2[:k], w3[:k]
	// Two four-lane streams per output leave room for Go's scheduled loads
	// without spilling accumulators. Equal lengths remove repeated bounds
	// checks in the vector body while keeping every access checked.
	var c00, c01 archsimd.Float32x4
	var c10, c11 archsimd.Float32x4
	var c20, c21 archsimd.Float32x4
	var c30, c31 archsimd.Float32x4
	p := 0
	for ; p+8 <= k; p += 8 {
		xv := (*[8]float32)(x[p : p+8])
		a0 := archsimd.LoadFloat32x4Array((*[4]float32)(xv[0:4]))
		a1 := archsimd.LoadFloat32x4Array((*[4]float32)(xv[4:8]))
		b0 := (*[8]float32)(w0[p : p+8])
		c00 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b0[0:4])), c00)
		c01 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b0[4:8])), c01)
		b1 := (*[8]float32)(w1[p : p+8])
		c10 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b1[0:4])), c10)
		c11 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b1[4:8])), c11)
		b2 := (*[8]float32)(w2[p : p+8])
		c20 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b2[0:4])), c20)
		c21 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b2[4:8])), c21)
		b3 := (*[8]float32)(w3[p : p+8])
		c30 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b3[0:4])), c30)
		c31 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b3[4:8])), c31)
	}
	s0 := vectorSum(c00.Add(c01))
	s1 := vectorSum(c10.Add(c11))
	s2 := vectorSum(c20.Add(c21))
	s3 := vectorSum(c30.Add(c31))
	for ; p < k; p++ {
		v := x[p]
		s0 += v * w0[p]
		s1 += v * w1[p]
		s2 += v * w2[p]
		s3 += v * w3[p]
	}
	_ = dst[3]
	dst[0], dst[1], dst[2], dst[3] = s0, s1, s2, s3
}

func vectorDot(x, w []float32) float32 {
	k := len(x)
	w = w[:k]
	var c0, c1, c2, c3 archsimd.Float32x4
	p := 0
	for ; p+16 <= k; p += 16 {
		a, b := (*[16]float32)(x[p:p+16]), (*[16]float32)(w[p:p+16])
		c0 = archsimd.LoadFloat32x4Array((*[4]float32)(a[0:4])).MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b[0:4])), c0)
		c1 = archsimd.LoadFloat32x4Array((*[4]float32)(a[4:8])).MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b[4:8])), c1)
		c2 = archsimd.LoadFloat32x4Array((*[4]float32)(a[8:12])).MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b[8:12])), c2)
		c3 = archsimd.LoadFloat32x4Array((*[4]float32)(a[12:16])).MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b[12:16])), c3)
	}
	sum := vectorSum(c0.Add(c1).Add(c2).Add(c3))
	for ; p < k; p++ {
		sum += x[p] * w[p]
	}
	return sum
}

func vectorSum(x archsimd.Float32x4) float32 {
	pairs := x.ConcatAddPairs(x)
	return pairs.ConcatAddPairs(pairs).GetElem(0)
}
