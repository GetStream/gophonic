// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && (amd64 || arm64)

package int8probe

import (
	"simd/archsimd"
	"unsafe"
)

func probeDot4x4(a0, a1, a2, a3, b0, b1, b2, b3 []float32, out *[16]float32) {
	n := len(a0)
	pa0 := unsafe.Pointer(unsafe.SliceData(a0))
	pa1 := unsafe.Pointer(unsafe.SliceData(a1))
	pa2 := unsafe.Pointer(unsafe.SliceData(a2))
	pa3 := unsafe.Pointer(unsafe.SliceData(a3))
	pb0 := unsafe.Pointer(unsafe.SliceData(b0))
	pb1 := unsafe.Pointer(unsafe.SliceData(b1))
	pb2 := unsafe.Pointer(unsafe.SliceData(b2))
	pb3 := unsafe.Pointer(unsafe.SliceData(b3))
	var c00, c01, c02, c03 archsimd.Float32x4
	var c10, c11, c12, c13 archsimd.Float32x4
	var c20, c21, c22, c23 archsimd.Float32x4
	var c30, c31, c32, c33 archsimd.Float32x4
	i := 0
	for ; i+4 <= n; i += 4 {
		x0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(pa0, uintptr(i*4))))
		x1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(pa1, uintptr(i*4))))
		x2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(pa2, uintptr(i*4))))
		x3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(pa3, uintptr(i*4))))
		y0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(pb0, uintptr(i*4))))
		y1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(pb1, uintptr(i*4))))
		y2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(pb2, uintptr(i*4))))
		y3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(pb3, uintptr(i*4))))
		c00 = x0.MulAdd(y0, c00)
		c01 = x0.MulAdd(y1, c01)
		c02 = x0.MulAdd(y2, c02)
		c03 = x0.MulAdd(y3, c03)
		c10 = x1.MulAdd(y0, c10)
		c11 = x1.MulAdd(y1, c11)
		c12 = x1.MulAdd(y2, c12)
		c13 = x1.MulAdd(y3, c13)
		c20 = x2.MulAdd(y0, c20)
		c21 = x2.MulAdd(y1, c21)
		c22 = x2.MulAdd(y2, c22)
		c23 = x2.MulAdd(y3, c23)
		c30 = x3.MulAdd(y0, c30)
		c31 = x3.MulAdd(y1, c31)
		c32 = x3.MulAdd(y2, c32)
		c33 = x3.MulAdd(y3, c33)
	}
	*out = [16]float32{
		probeReduce(c00), probeReduce(c01), probeReduce(c02), probeReduce(c03),
		probeReduce(c10), probeReduce(c11), probeReduce(c12), probeReduce(c13),
		probeReduce(c20), probeReduce(c21), probeReduce(c22), probeReduce(c23),
		probeReduce(c30), probeReduce(c31), probeReduce(c32), probeReduce(c33),
	}
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

func probeReduce(value archsimd.Float32x4) float32 {
	var lanes [4]float32
	value.StoreArray(&lanes)
	return lanes[0] + lanes[1] + lanes[2] + lanes[3]
}
