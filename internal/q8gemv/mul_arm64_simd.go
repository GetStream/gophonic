// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package q8gemv

import (
	"simd/archsimd"
	"unsafe"
)

// mulKernel converts packed int8 weights directly to float32 vectors and
// accumulates against the float32 activations. Four independent FMLA chains
// hide the latency of the NEON multiply-accumulate instruction. A row is read
// once and is never materialized as float32 scratch.
func mulKernel(x []float32, q []int8, scales, dst []float32, k, n int) {
	if n == 0 {
		return
	}
	xBase := unsafe.Pointer(unsafe.SliceData(x))
	qBase := unsafe.Pointer(unsafe.SliceData(q))
	for row := range n {
		var c0, c1, c2, c3 archsimd.Float32x4
		qRow := unsafe.Add(qBase, uintptr(row*k))
		for i := 0; i+16 <= k; i += 16 {
			packed := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Add(qRow, uintptr(i))))
			low := packed.ExtendLo8ToInt16()
			high := packed.HiToLo().ExtendLo8ToInt16()
			w0 := low.ExtendLo4ToInt32().ConvertToFloat32()
			w1 := low.HiToLo().ExtendLo4ToInt32().ConvertToFloat32()
			w2 := high.ExtendLo4ToInt32().ConvertToFloat32()
			w3 := high.HiToLo().ExtendLo4ToInt32().ConvertToFloat32()

			a0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xBase, uintptr(i*4))))
			a1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xBase, uintptr((i+4)*4))))
			a2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xBase, uintptr((i+8)*4))))
			a3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xBase, uintptr((i+12)*4))))
			c0 = a0.MulAdd(w0, c0)
			c1 = a1.MulAdd(w1, c1)
			c2 = a2.MulAdd(w2, c2)
			c3 = a3.MulAdd(w3, c3)
		}

		lanes := c0.Add(c1).Add(c2).Add(c3)
		sum := lanes.GetElem(0) + lanes.GetElem(1) + lanes.GetElem(2) + lanes.GetElem(3)
		rowStart := k &^ 15
		for col := rowStart; col < k; col++ {
			sum += x[col] * float32(q[row*k+col])
		}
		dst[row] = sum * scales[row]
	}
}
