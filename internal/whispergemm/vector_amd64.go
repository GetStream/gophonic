// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whispergemm

import "simd/archsimd"

// Four rows share each input load. Two independent eight-wide accumulators per
// row cover sixteen K values per loop, fitting within the amd64 vector file.
func mulVector(dst, weights []float32, stride int, x []float32, rows int) {
	r := 0
	for ; r+4 <= rows; r += 4 {
		vector4AMD64(dst[r:r+4], x,
			weights[r*stride:r*stride+len(x)], weights[(r+1)*stride:(r+1)*stride+len(x)],
			weights[(r+2)*stride:(r+2)*stride+len(x)], weights[(r+3)*stride:(r+3)*stride+len(x)])
	}
	if r < rows {
		mulVectorScalar(dst[r:rows], weights[r*stride:], stride, x, rows-r)
	}
}

//go:nosplit
func vector4AMD64(dst, x, w0, w1, w2, w3 []float32) {
	var c00, c01, c10, c11, c20, c21, c30, c31 archsimd.Float32x8
	p := 0
	for ; p+16 <= len(x); p += 16 {
		a0 := archsimd.LoadFloat32x8(x[p:])
		a1 := archsimd.LoadFloat32x8(x[p+8:])
		c00 = a0.MulAdd(archsimd.LoadFloat32x8(w0[p:]), c00)
		c01 = a1.MulAdd(archsimd.LoadFloat32x8(w0[p+8:]), c01)
		c10 = a0.MulAdd(archsimd.LoadFloat32x8(w1[p:]), c10)
		c11 = a1.MulAdd(archsimd.LoadFloat32x8(w1[p+8:]), c11)
		c20 = a0.MulAdd(archsimd.LoadFloat32x8(w2[p:]), c20)
		c21 = a1.MulAdd(archsimd.LoadFloat32x8(w2[p+8:]), c21)
		c30 = a0.MulAdd(archsimd.LoadFloat32x8(w3[p:]), c30)
		c31 = a1.MulAdd(archsimd.LoadFloat32x8(w3[p+8:]), c31)
	}
	s0 := reducePairAMD64(c00, c01)
	s1 := reducePairAMD64(c10, c11)
	s2 := reducePairAMD64(c20, c21)
	s3 := reducePairAMD64(c30, c31)
	for ; p < len(x); p++ {
		v := x[p]
		s0 += v * w0[p]
		s1 += v * w1[p]
		s2 += v * w2[p]
		s3 += v * w3[p]
	}
	dst[0], dst[1], dst[2], dst[3] = s0, s1, s2, s3
}

func reducePairAMD64(x, y archsimd.Float32x8) float32 {
	z := x.Add(y)
	halves := z.GetLo().Add(z.GetHi())
	var lanes [4]float32
	halves.StoreArray(&lanes)
	return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}
