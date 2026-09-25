// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package vec

import (
	"simd/archsimd"
	"unsafe"
)

func Dot4(a, b0, b1, b2, b3 []float32) (float32, float32, float32, float32) {
	n := len(a)
	ap := unsafe.Pointer(unsafe.SliceData(a))
	b0p := unsafe.Pointer(unsafe.SliceData(b0))
	b1p := unsafe.Pointer(unsafe.SliceData(b1))
	b2p := unsafe.Pointer(unsafe.SliceData(b2))
	b3p := unsafe.Pointer(unsafe.SliceData(b3))
	var a0, a1, a2, a3 archsimd.Float32x4
	var c00, c01, c02, c03 archsimd.Float32x4
	var c10, c11, c12, c13 archsimd.Float32x4
	var c20, c21, c22, c23 archsimd.Float32x4
	var c30, c31, c32, c33 archsimd.Float32x4
	i := 0
	for ; i+16 <= n; i += 16 {
		a0 = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap, uintptr(i*4))))
		a1 = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap, uintptr((i+4)*4))))
		a2 = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap, uintptr((i+8)*4))))
		a3 = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap, uintptr((i+12)*4))))
		c00 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b0p, uintptr(i*4)))), c00)
		c01 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b0p, uintptr((i+4)*4)))), c01)
		c02 = a2.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b0p, uintptr((i+8)*4)))), c02)
		c03 = a3.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b0p, uintptr((i+12)*4)))), c03)
		c10 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b1p, uintptr(i*4)))), c10)
		c11 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b1p, uintptr((i+4)*4)))), c11)
		c12 = a2.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b1p, uintptr((i+8)*4)))), c12)
		c13 = a3.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b1p, uintptr((i+12)*4)))), c13)
		c20 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b2p, uintptr(i*4)))), c20)
		c21 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b2p, uintptr((i+4)*4)))), c21)
		c22 = a2.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b2p, uintptr((i+8)*4)))), c22)
		c23 = a3.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b2p, uintptr((i+12)*4)))), c23)
		c30 = a0.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b3p, uintptr(i*4)))), c30)
		c31 = a1.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b3p, uintptr((i+4)*4)))), c31)
		c32 = a2.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b3p, uintptr((i+8)*4)))), c32)
		c33 = a3.MulAdd(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(b3p, uintptr((i+12)*4)))), c33)
	}
	s0 := Reduce4(c00, c01, c02, c03)
	s1 := Reduce4(c10, c11, c12, c13)
	s2 := Reduce4(c20, c21, c22, c23)
	s3 := Reduce4(c30, c31, c32, c33)
	for ; i < n; i++ {
		x := a[i]
		s0 += x * b0[i]
		s1 += x * b1[i]
		s2 += x * b2[i]
		s3 += x * b3[i]
	}
	return s0, s1, s2, s3
}

func Reduce4(a, b, c, d archsimd.Float32x4) float32 {
	lanes := a.Add(b).Add(c).Add(d)
	return lanes.GetElem(0) + lanes.GetElem(1) + lanes.GetElem(2) + lanes.GetElem(3)
}
