// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && (amd64 || arm64)

package gophonic

import "simd/archsimd"
import "unsafe"

func dotProduct(a, b []float32) float32 {
	n := len(a)
	aPtr := unsafe.Pointer(unsafe.SliceData(a))
	bPtr := unsafe.Pointer(unsafe.SliceData(b))
	var acc0, acc1, acc2, acc3 archsimd.Float32x4
	i := 0
	for ; i+16 <= n; i += 16 {
		a0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(aPtr, uintptr(i*4))))
		b0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bPtr, uintptr(i*4))))
		a1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(aPtr, uintptr((i+4)*4))))
		b1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bPtr, uintptr((i+4)*4))))
		a2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(aPtr, uintptr((i+8)*4))))
		b2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bPtr, uintptr((i+8)*4))))
		a3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(aPtr, uintptr((i+12)*4))))
		b3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bPtr, uintptr((i+12)*4))))
		acc0 = a0.MulAdd(b0, acc0)
		acc1 = a1.MulAdd(b1, acc1)
		acc2 = a2.MulAdd(b2, acc2)
		acc3 = a3.MulAdd(b3, acc3)
	}
	acc := acc0.Add(acc1).Add(acc2).Add(acc3)
	for ; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(aPtr, uintptr(i*4))))
		y := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bPtr, uintptr(i*4))))
		acc = x.MulAdd(y, acc)
	}
	var lanes [4]float32
	acc.StoreArray(&lanes)
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3]
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
