// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package whispergemm

import "simd/archsimd"

func mulVector(dst, weights []float32, stride int, x []float32, rows int) {
	r := 0
	// Long reductions and strided attention heads benefit from sharing the
	// input across eight outputs. Four output streams suit the large contiguous
	// vocabulary matrix, whose weights are read once per generated token.
	if len(x) >= 512 || stride > len(x) {
		for ; r+8 <= rows; r += 8 {
			vector8(dst[r:r+8], x,
				weights[r*stride:r*stride+len(x)], weights[(r+1)*stride:(r+1)*stride+len(x)],
				weights[(r+2)*stride:(r+2)*stride+len(x)], weights[(r+3)*stride:(r+3)*stride+len(x)],
				weights[(r+4)*stride:(r+4)*stride+len(x)], weights[(r+5)*stride:(r+5)*stride+len(x)],
				weights[(r+6)*stride:(r+6)*stride+len(x)], weights[(r+7)*stride:(r+7)*stride+len(x)])
		}
	}
	for ; r+4 <= rows; r += 4 {
		vector4(dst[r:r+4], x,
			weights[r*stride:r*stride+len(x)], weights[(r+1)*stride:(r+1)*stride+len(x)],
			weights[(r+2)*stride:(r+2)*stride+len(x)], weights[(r+3)*stride:(r+3)*stride+len(x)])
	}
	for ; r < rows; r++ {
		dst[r] = vectorDot(x, weights[r*stride:r*stride+len(x)])
	}
}

// vector4 keeps two reduction streams per output and unrolls the reduction by
// sixteen values, amortizing the loop and row-slice checks over twice the work.
func vector4(dst, x, w0, w1, w2, w3 []float32) {
	k := len(x)
	w0 = w0[:k]
	w1 = w1[:k]
	w2 = w2[:k]
	w3 = w3[:k]
	var c00, c01, c10, c11, c20, c21, c30, c31 archsimd.Float32x4
	p := 0
	for ; p <= k-16; p += 16 {
		xv := (*[16]float32)(x[p : p+16])
		a0 := archsimd.LoadFloat32x4Array((*[4]float32)(xv[0:4]))
		a1 := archsimd.LoadFloat32x4Array((*[4]float32)(xv[4:8]))
		a2 := archsimd.LoadFloat32x4Array((*[4]float32)(xv[8:12]))
		a3 := archsimd.LoadFloat32x4Array((*[4]float32)(xv[12:16]))
		b0 := (*[16]float32)(w0[p : p+16])
		c00 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b0[0:4])), c00)
		c01 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b0[4:8])), c01)
		c00 = a2.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b0[8:12])), c00)
		c01 = a3.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b0[12:16])), c01)
		b1 := (*[16]float32)(w1[p : p+16])
		c10 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b1[0:4])), c10)
		c11 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b1[4:8])), c11)
		c10 = a2.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b1[8:12])), c10)
		c11 = a3.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b1[12:16])), c11)
		b2 := (*[16]float32)(w2[p : p+16])
		c20 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b2[0:4])), c20)
		c21 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b2[4:8])), c21)
		c20 = a2.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b2[8:12])), c20)
		c21 = a3.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b2[12:16])), c21)
		b3 := (*[16]float32)(w3[p : p+16])
		c30 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b3[0:4])), c30)
		c31 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b3[4:8])), c31)
		c30 = a2.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b3[8:12])), c30)
		c31 = a3.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b3[12:16])), c31)
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
	dst[0] = s0
	dst[1] = s1
	dst[2] = s2
	dst[3] = s3
}

// vector8 reuses each input load across eight rows, with two independent
// reduction streams per row. Keep row slices independent here: bounding all
// capacities to k makes Go 1.27 schedule every row load before its FMA and
// spill vector registers. The checked slices below generate no vector spills.
func vector8(dst, x, w0, w1, w2, w3, w4, w5, w6, w7 []float32) {
	k := len(x)
	w0 = w0[:k]
	w1 = w1[:k]
	w2 = w2[:k]
	w3 = w3[:k]
	w4 = w4[:k]
	w5 = w5[:k]
	w6 = w6[:k]
	w7 = w7[:k]
	var c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71 archsimd.Float32x4
	p := 0
	for ; p <= k-8; p += 8 {
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
		b4 := (*[8]float32)(w4[p : p+8])
		c40 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b4[0:4])), c40)
		c41 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b4[4:8])), c41)
		b5 := (*[8]float32)(w5[p : p+8])
		c50 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b5[0:4])), c50)
		c51 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b5[4:8])), c51)
		b6 := (*[8]float32)(w6[p : p+8])
		c60 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b6[0:4])), c60)
		c61 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b6[4:8])), c61)
		b7 := (*[8]float32)(w7[p : p+8])
		c70 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b7[0:4])), c70)
		c71 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(b7[4:8])), c71)
	}
	s0 := vectorSum(c00.Add(c01))
	s1 := vectorSum(c10.Add(c11))
	s2 := vectorSum(c20.Add(c21))
	s3 := vectorSum(c30.Add(c31))
	s4 := vectorSum(c40.Add(c41))
	s5 := vectorSum(c50.Add(c51))
	s6 := vectorSum(c60.Add(c61))
	s7 := vectorSum(c70.Add(c71))
	for ; p < k; p++ {
		v := x[p]
		s0 += v * w0[p]
		s1 += v * w1[p]
		s2 += v * w2[p]
		s3 += v * w3[p]
		s4 += v * w4[p]
		s5 += v * w5[p]
		s6 += v * w6[p]
		s7 += v * w7[p]
	}
	_ = dst[7]
	dst[0] = s0
	dst[1] = s1
	dst[2] = s2
	dst[3] = s3
	dst[4] = s4
	dst[5] = s5
	dst[6] = s6
	dst[7] = s7
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
