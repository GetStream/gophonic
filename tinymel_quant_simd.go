// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gofloor

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

// quantizeTinySIMD preserves DynamicQuantizeLinear's float32 division and
// ties-to-even rounding. It does not replace division with a reciprocal.
// Non-finite ranges use the scalar implementation's existing behavior.
func quantizeTinySIMD(input []float32, output []uint8) tinyQuantParams {
	var min0, min1, max0, max1 archsimd.Float32x4
	ip := unsafe.Pointer(unsafe.SliceData(input))
	op := unsafe.Pointer(unsafe.SliceData(output))
	n := len(input)
	if len(output) < n {
		return quantizeTinyScalar(input, output)
	}
	i := 0
	for ; i+8 <= n; i += 8 {
		x0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ip, uintptr(i*4))))
		x1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ip, uintptr((i+4)*4))))
		min0 = min0.Min(x0)
		min1 = min1.Min(x1)
		max0 = max0.Max(x0)
		max1 = max1.Max(x1)
	}
	minValue, maxValue := min0.Min(min1).ReduceMin(), max0.Max(max1).ReduceMax()
	for ; i < n; i++ {
		value := input[i]
		if math.IsNaN(float64(value)) {
			return quantizeTinyScalar(input, output)
		}
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
	}
	if math.IsNaN(float64(minValue)) || math.IsNaN(float64(maxValue)) || math.IsInf(float64(minValue), 0) || math.IsInf(float64(maxValue), 0) {
		return quantizeTinyScalar(input, output)
	}
	scale := (maxValue - minValue) / 255
	if scale == 0 {
		scale = 1
	}
	zeroFloat := float32(math.RoundToEven(float64(-minValue / scale)))
	if zeroFloat < 0 {
		zeroFloat = 0
	} else if zeroFloat > 255 {
		zeroFloat = 255
	}
	zero := uint8(zeroFloat)
	scale4 := archsimd.BroadcastFloat32x4(scale)
	zero4 := archsimd.BroadcastFloat32x4(float32(zero))
	i = 0
	for ; i+16 <= n; i += 16 {
		x0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ip, uintptr(i*4))))
		x1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ip, uintptr((i+4)*4))))
		x2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ip, uintptr((i+8)*4))))
		x3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(ip, uintptr((i+12)*4))))
		q0 := x0.Div(scale4).Round().Add(zero4).ConvertToInt32().SaturateToUint16()
		q1 := x1.Div(scale4).Round().Add(zero4).ConvertToInt32().SaturateToUint16()
		q2 := x2.Div(scale4).Round().Add(zero4).ConvertToInt32().SaturateToUint16()
		q3 := x3.Div(scale4).Round().Add(zero4).ConvertToInt32().SaturateToUint16()
		q01 := q0.ReshapeToUint64s().ConcatEven(q1.ReshapeToUint64s()).ReshapeToUint16s().SaturateToUint8()
		q23 := q2.ReshapeToUint64s().ConcatEven(q3.ReshapeToUint64s()).ReshapeToUint16s().SaturateToUint8()
		q01.ReshapeToUint64s().ConcatEven(q23.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(op, uintptr(i))))
	}
	for ; i < n; i++ {
		q := float32(math.RoundToEven(float64(input[i]/scale))) + float32(zero)
		if q < 0 {
			q = 0
		} else if q > 255 {
			q = 255
		}
		output[i] = uint8(q)
	}
	return tinyQuantParams{scale: scale, zero: zero}
}
