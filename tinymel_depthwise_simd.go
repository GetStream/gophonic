// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gophonic

import (
	"runtime"
	"simd/archsimd"
	"unsafe"
)

// runTinyDepthwiseSIMDRange evaluates sixteen independent channels per vector
// tile. packed must come from packTinyDepthwiseWeights for these same weights.
// Padding has the activation zero point, so omitted taps contribute zero.
func runTinyDepthwiseSIMDRange(conv tinyConv1D, packed []uint8, input []uint8, params tinyQuantParams, output []float32, inputLength, fromT, toT int) int {
	if runtime.GOARCH != "arm64" || conv.groups != conv.inChannels || conv.outChannels != conv.inChannels || conv.inChannels%16 != 0 || len(packed) != conv.kernel*conv.inChannels {
		return runTinyConvSIMDRange(conv, input, params, output, inputLength, fromT, toT)
	}
	pad := conv.kernel / 2
	outputLength := (inputLength+2*pad-conv.kernel)/conv.stride + 1
	if fromT < 0 {
		fromT = 0
	}
	if toT > outputLength {
		toT = outputLength
	}
	scale := archsimd.BroadcastFloat32x4(params.scale * conv.weightScale)
	zero := archsimd.BroadcastUint16x8(uint16(params.zero))
	wzero := archsimd.BroadcastUint16x8(uint16(conv.weightZero))
	ip := unsafe.Pointer(unsafe.SliceData(input))
	wp := unsafe.Pointer(unsafe.SliceData(packed))
	bp := unsafe.Pointer(unsafe.SliceData(conv.bias))
	op := unsafe.Pointer(unsafe.SliceData(output))
	for t := fromT; t < toT; t++ {
		for c := 0; c < conv.inChannels; c += 16 {
			var a0, a1, a2, a3 archsimd.Int32x4
			for k := 0; k < conv.kernel; k++ {
				it := t*conv.stride + k - pad
				if it < 0 || it >= inputLength {
					continue
				}
				x := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(ip, uintptr(it*conv.inChannels+c))))
				w := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(k*conv.inChannels+c))))
				xl := x.ExtendLo8ToUint16().Sub(zero).BitsToInt16()
				xh := x.HiToLo().ExtendLo8ToUint16().Sub(zero).BitsToInt16()
				wl := w.ExtendLo8ToUint16().Sub(wzero).BitsToInt16()
				wh := w.HiToLo().ExtendLo8ToUint16().Sub(wzero).BitsToInt16()
				a0 = a0.Add(xl.MulWidenLo(wl))
				a1 = a1.Add(xl.HiToLo().MulWidenLo(wl.HiToLo()))
				a2 = a2.Add(xh.MulWidenLo(wh))
				a3 = a3.Add(xh.HiToLo().MulWidenLo(wh.HiToLo()))
			}
			y0 := tinyConvDequantizeVector(a0, scale, archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp, uintptr(c*4)))))
			y1 := tinyConvDequantizeVector(a1, scale, archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp, uintptr((c+4)*4)))))
			y2 := tinyConvDequantizeVector(a2, scale, archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp, uintptr((c+8)*4)))))
			y3 := tinyConvDequantizeVector(a3, scale, archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(bp, uintptr((c+12)*4)))))
			base := t*conv.inChannels + c
			y0.StoreArray((*[4]float32)(unsafe.Add(op, uintptr(base*4))))
			y1.StoreArray((*[4]float32)(unsafe.Add(op, uintptr((base+4)*4))))
			y2.StoreArray((*[4]float32)(unsafe.Add(op, uintptr((base+8)*4))))
			y3.StoreArray((*[4]float32)(unsafe.Add(op, uintptr((base+12)*4))))
		}
	}
	return outputLength
}

// Match the scalar compiler's rounding: arm64 fuses the multiply and bias,
// whereas the baseline amd64 instruction set emits separate operations.
func tinyConvDequantizeVector(acc archsimd.Int32x4, scale, bias archsimd.Float32x4) archsimd.Float32x4 {
	if runtime.GOARCH == "arm64" {
		return acc.ConvertToFloat32().MulAdd(scale, bias)
	}
	return acc.ConvertToFloat32().Mul(scale).Add(bias)
}
