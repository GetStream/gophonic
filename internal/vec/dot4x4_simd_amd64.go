// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64

package vec

import "simd/archsimd"

// Dot4x4 is the amd64 counterpart of dot4x4_simd.go. AVX2 offers only
// 16 vector registers, so the 4x4 Float32x4 tile (16 live accumulators plus
// 8 sources) spills vector registers to the stack on every iteration. This
// version uses Float32x8 (one VFMADD213PS per multiply-add) with a 4x2 tile:
// 8 live accumulators plus 6 sources fit comfortably in registers. The second
// b-pair runs as a second pass; the re-read a-vectors stay hot in cache.
func Dot4x4(a0, a1, a2, a3, b0, b1, b2, b3 []float32, out *[16]float32) {
	n := len(a0)
	var c00, c01 archsimd.Float32x8
	var c10, c11 archsimd.Float32x8
	var c20, c21 archsimd.Float32x8
	var c30, c31 archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		x0 := archsimd.LoadFloat32x8(a0[i:])
		x1 := archsimd.LoadFloat32x8(a1[i:])
		x2 := archsimd.LoadFloat32x8(a2[i:])
		x3 := archsimd.LoadFloat32x8(a3[i:])
		y0 := archsimd.LoadFloat32x8(b0[i:])
		y1 := archsimd.LoadFloat32x8(b1[i:])
		c00 = x0.MulAdd(y0, c00)
		c01 = x0.MulAdd(y1, c01)
		c10 = x1.MulAdd(y0, c10)
		c11 = x1.MulAdd(y1, c11)
		c20 = x2.MulAdd(y0, c20)
		c21 = x2.MulAdd(y1, c21)
		c30 = x3.MulAdd(y0, c30)
		c31 = x3.MulAdd(y1, c31)
	}
	out[0], out[1] = reduceVector8(c00), reduceVector8(c01)
	out[4], out[5] = reduceVector8(c10), reduceVector8(c11)
	out[8], out[9] = reduceVector8(c20), reduceVector8(c21)
	out[12], out[13] = reduceVector8(c30), reduceVector8(c31)

	var c02, c03 archsimd.Float32x8
	var c12, c13 archsimd.Float32x8
	var c22, c23 archsimd.Float32x8
	var c32, c33 archsimd.Float32x8
	i = 0
	for ; i+8 <= n; i += 8 {
		x0 := archsimd.LoadFloat32x8(a0[i:])
		x1 := archsimd.LoadFloat32x8(a1[i:])
		x2 := archsimd.LoadFloat32x8(a2[i:])
		x3 := archsimd.LoadFloat32x8(a3[i:])
		y2 := archsimd.LoadFloat32x8(b2[i:])
		y3 := archsimd.LoadFloat32x8(b3[i:])
		c02 = x0.MulAdd(y2, c02)
		c03 = x0.MulAdd(y3, c03)
		c12 = x1.MulAdd(y2, c12)
		c13 = x1.MulAdd(y3, c13)
		c22 = x2.MulAdd(y2, c22)
		c23 = x2.MulAdd(y3, c23)
		c32 = x3.MulAdd(y2, c32)
		c33 = x3.MulAdd(y3, c33)
	}
	out[2], out[3] = reduceVector8(c02), reduceVector8(c03)
	out[6], out[7] = reduceVector8(c12), reduceVector8(c13)
	out[10], out[11] = reduceVector8(c22), reduceVector8(c23)
	out[14], out[15] = reduceVector8(c32), reduceVector8(c33)

	for ; i < n; i++ {
		x0, x1, x2, x3 := a0[i], a1[i], a2[i], a3[i]
		y0, y1, y2, y3 := b0[i], b1[i], b2[i], b3[i]
		out[0] += x0 * y0
		out[1] += x0 * y1
		out[2] += x0 * y2
		out[3] += x0 * y3
		out[4] += x1 * y0
		out[5] += x1 * y1
		out[6] += x1 * y2
		out[7] += x1 * y3
		out[8] += x2 * y0
		out[9] += x2 * y1
		out[10] += x2 * y2
		out[11] += x2 * y3
		out[12] += x3 * y0
		out[13] += x3 * y1
		out[14] += x3 * y2
		out[15] += x3 * y3
	}
}

// reduceVector8 folds the two 128-bit halves before touching memory so only
// four lanes reach the stack instead of eight.
func reduceVector8(value archsimd.Float32x8) float32 {
	halves := value.GetLo().Add(value.GetHi())
	var lanes [4]float32
	halves.StoreArray(&lanes)
	return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}
