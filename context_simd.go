// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && (amd64 || arm64)

package gophonic

import (
	"simd/archsimd"
	"unsafe"
)

func attentionContext4(scores0, scores1, scores2, scores3, values, output0, output1, output2, output3 []float32, headOffset int) {
	valueBase := unsafe.Pointer(unsafe.SliceData(values))
	score0 := unsafe.Pointer(unsafe.SliceData(scores0))
	score1 := unsafe.Pointer(unsafe.SliceData(scores1))
	score2 := unsafe.Pointer(unsafe.SliceData(scores2))
	score3 := unsafe.Pointer(unsafe.SliceData(scores3))
	out0 := unsafe.Pointer(unsafe.SliceData(output0))
	out1 := unsafe.Pointer(unsafe.SliceData(output1))
	out2 := unsafe.Pointer(unsafe.SliceData(output2))
	out3 := unsafe.Pointer(unsafe.SliceData(output3))
	headBytes := uintptr(headOffset * 4)
	rowBytes := uintptr(hiddenSize * 4)
	for d := 0; d < headSize; d += 4 {
		var acc0, acc1, acc2, acc3 archsimd.Float32x4
		valueOffset := headBytes + uintptr(d*4)
		for key := 0; key < sequenceLength; key++ {
			value := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(valueBase, uintptr(key)*rowBytes+valueOffset)))
			weight0 := *(*float32)(unsafe.Add(score0, uintptr(key*4)))
			weight1 := *(*float32)(unsafe.Add(score1, uintptr(key*4)))
			weight2 := *(*float32)(unsafe.Add(score2, uintptr(key*4)))
			weight3 := *(*float32)(unsafe.Add(score3, uintptr(key*4)))
			acc0 = archsimd.BroadcastFloat32x4(weight0).MulAdd(value, acc0)
			acc1 = archsimd.BroadcastFloat32x4(weight1).MulAdd(value, acc1)
			acc2 = archsimd.BroadcastFloat32x4(weight2).MulAdd(value, acc2)
			acc3 = archsimd.BroadcastFloat32x4(weight3).MulAdd(value, acc3)
		}
		acc0.StoreArray((*[4]float32)(unsafe.Add(out0, uintptr(d*4))))
		acc1.StoreArray((*[4]float32)(unsafe.Add(out1, uintptr(d*4))))
		acc2.StoreArray((*[4]float32)(unsafe.Add(out2, uintptr(d*4))))
		acc3.StoreArray((*[4]float32)(unsafe.Add(out3, uintptr(d*4))))
	}
}
