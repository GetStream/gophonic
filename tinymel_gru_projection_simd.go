// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gophonic

import (
	"simd/archsimd"
	"unsafe"
)

// tinyGRUProjectInputs batches adjacent timesteps while preserving the four
// accumulation streams and final reduction order used by dotProduct4. Weight
// rows are outermost so the same two rows stay hot while traversing all input
// frames, instead of streaming the full matrix again for each timestep pair.
func tinyGRUProjectInputs(input, weights, projected []float32, sequenceLength, inputSize, outputSize int) {
	// The reference divides outputs into three gates. Keep each gate's scalar
	// remainder behavior for dimensions that do not fit complete four-row dots.
	if outputSize%12 != 0 {
		tinyGRUProjectInputsRowwise(input, weights, projected, sequenceLength, inputSize, outputSize)
		return
	}
	tiledTimes := sequenceLength &^ 1
	for o := 0; o < outputSize; o += 2 {
		w0 := weights[o*inputSize : (o+1)*inputSize]
		w1 := weights[(o+1)*inputSize : (o+2)*inputSize]
		for t := 0; t < tiledTimes; t += 2 {
			x0 := input[t*inputSize : (t+1)*inputSize]
			x1 := input[(t+1)*inputSize : (t+2)*inputSize]
			p00, p01, p10, p11 := dotProduct2x2Strict(x0, x1, w0, w1)
			projected[t*outputSize+o] = p00
			projected[t*outputSize+o+1] = p01
			projected[(t+1)*outputSize+o] = p10
			projected[(t+1)*outputSize+o+1] = p11
		}
	}
	if tiledTimes < sequenceLength {
		tinyGRUProjectInputsRowwise(input[tiledTimes*inputSize:], weights, projected[tiledTimes*outputSize:], sequenceLength-tiledTimes, inputSize, outputSize)
	}
}

// dotProduct2x2Strict computes two input rows against two weight rows. Each
// result performs precisely dotProduct4's FMA and reduction sequence.
func dotProduct2x2Strict(a0, a1, b0, b1 []float32) (float32, float32, float32, float32) {
	n := len(a0)
	ap0 := unsafe.Pointer(unsafe.SliceData(a0))
	ap1 := unsafe.Pointer(unsafe.SliceData(a1))
	bp0 := unsafe.Pointer(unsafe.SliceData(b0))
	bp1 := unsafe.Pointer(unsafe.SliceData(b1))
	var c000, c001, c002, c003 archsimd.Float32x4
	var c010, c011, c012, c013 archsimd.Float32x4
	var c100, c101, c102, c103 archsimd.Float32x4
	var c110, c111, c112, c113 archsimd.Float32x4
	i := 0
	for ; i+16 <= n; i += 16 {
		x00 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap0, uintptr((i+0)*4))))
		x10 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap1, uintptr((i+0)*4))))
		w00 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp0, uintptr((i+0)*4))))
		w10 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp1, uintptr((i+0)*4))))
		c000 = x00.MulAdd(w00, c000)
		c010 = x00.MulAdd(w10, c010)
		c100 = x10.MulAdd(w00, c100)
		c110 = x10.MulAdd(w10, c110)
		x01 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap0, uintptr((i+4)*4))))
		x11 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap1, uintptr((i+4)*4))))
		w01 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp0, uintptr((i+4)*4))))
		w11 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp1, uintptr((i+4)*4))))
		c001 = x01.MulAdd(w01, c001)
		c011 = x01.MulAdd(w11, c011)
		c101 = x11.MulAdd(w01, c101)
		c111 = x11.MulAdd(w11, c111)
		x02 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap0, uintptr((i+8)*4))))
		x12 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap1, uintptr((i+8)*4))))
		w02 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp0, uintptr((i+8)*4))))
		w12 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp1, uintptr((i+8)*4))))
		c002 = x02.MulAdd(w02, c002)
		c012 = x02.MulAdd(w12, c012)
		c102 = x12.MulAdd(w02, c102)
		c112 = x12.MulAdd(w12, c112)
		x03 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap0, uintptr((i+12)*4))))
		x13 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ap1, uintptr((i+12)*4))))
		w03 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp0, uintptr((i+12)*4))))
		w13 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp1, uintptr((i+12)*4))))
		c003 = x03.MulAdd(w03, c003)
		c013 = x03.MulAdd(w13, c013)
		c103 = x13.MulAdd(w03, c103)
		c113 = x13.MulAdd(w13, c113)
	}
	s00 := reduce4(c000, c001, c002, c003)
	s01 := reduce4(c010, c011, c012, c013)
	s10 := reduce4(c100, c101, c102, c103)
	s11 := reduce4(c110, c111, c112, c113)
	for ; i < n; i++ {
		x0, x1, w0, w1 := a0[i], a1[i], b0[i], b1[i]
		s00 += x0 * w0
		s01 += x0 * w1
		s10 += x1 * w0
		s11 += x1 * w1
	}
	return s00, s01, s10, s11
}
