// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package tinymel

import (
	"simd/archsimd"
	"unsafe"
)

// runTinyConvTile4Range computes four output channels together. Each input
// vector is widened and centered once, then reused by four independent dots.
// Integer additions are associative modulo 2^32, matching ConvInteger exactly.
func runTinyConvTile4Range(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength, fromT, toT int) int {
	if conv.groups != 1 || conv.inChannels%16 != 0 || conv.outChannels%4 != 0 || len(conv.packed) != conv.outChannels*conv.kernel*conv.inChannels {
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
	scale := params.scale * conv.weightScale
	zero16 := archsimd.BroadcastUint16x8(uint16(params.zero))
	wzero16 := archsimd.BroadcastUint16x8(uint16(conv.weightZero))
	inputData := unsafe.Pointer(unsafe.SliceData(input))
	weightData := unsafe.Pointer(unsafe.SliceData(conv.packed))
	weightStride := conv.kernel * conv.inChannels
	for t := fromT; t < toT; t++ {
		for oc := 0; oc < conv.outChannels; oc += 4 {
			var c0l, c0h archsimd.Int32x4
			var c1l, c1h archsimd.Int32x4
			var c2l, c2h archsimd.Int32x4
			var c3l, c3h archsimd.Int32x4
			for k := 0; k < conv.kernel; k++ {
				it := t*conv.stride + k - pad
				if it < 0 || it >= inputLength {
					continue
				}
				ip := unsafe.Add(inputData, uintptr(it*conv.inChannels))
				wp := unsafe.Add(weightData, uintptr(oc*weightStride+k*conv.inChannels))
				for ic := 0; ic < conv.inChannels; ic += 16 {
					x := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(ip, uintptr(ic))))
					xl := x.ExtendLo8ToUint16().Sub(zero16).BitsToInt16()
					xh := x.HiToLo().ExtendLo8ToUint16().Sub(zero16).BitsToInt16()
					w0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(0*weightStride+ic))))
					w0l := w0.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					w0h := w0.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					c0l = c0l.Add(xl.MulWidenLo(w0l))
					c0h = c0h.Add(xl.HiToLo().MulWidenLo(w0l.HiToLo()))
					c0l = c0l.Add(xh.MulWidenLo(w0h))
					c0h = c0h.Add(xh.HiToLo().MulWidenLo(w0h.HiToLo()))
					w1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(1*weightStride+ic))))
					w1l := w1.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					w1h := w1.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					c1l = c1l.Add(xl.MulWidenLo(w1l))
					c1h = c1h.Add(xl.HiToLo().MulWidenLo(w1l.HiToLo()))
					c1l = c1l.Add(xh.MulWidenLo(w1h))
					c1h = c1h.Add(xh.HiToLo().MulWidenLo(w1h.HiToLo()))
					w2 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(2*weightStride+ic))))
					w2l := w2.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					w2h := w2.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					c2l = c2l.Add(xl.MulWidenLo(w2l))
					c2h = c2h.Add(xl.HiToLo().MulWidenLo(w2l.HiToLo()))
					c2l = c2l.Add(xh.MulWidenLo(w2h))
					c2h = c2h.Add(xh.HiToLo().MulWidenLo(w2h.HiToLo()))
					w3 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(3*weightStride+ic))))
					w3l := w3.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					w3h := w3.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					c3l = c3l.Add(xl.MulWidenLo(w3l))
					c3h = c3h.Add(xl.HiToLo().MulWidenLo(w3l.HiToLo()))
					c3l = c3l.Add(xh.MulWidenLo(w3h))
					c3h = c3h.Add(xh.HiToLo().MulWidenLo(w3h.HiToLo()))
				}
			}
			v0 := float32(c0l.Add(c0h).ReduceSum()) * scale
			output[t*conv.outChannels+oc+0] = v0 + conv.bias[oc+0]
			v1 := float32(c1l.Add(c1h).ReduceSum()) * scale
			output[t*conv.outChannels+oc+1] = v1 + conv.bias[oc+1]
			v2 := float32(c2l.Add(c2h).ReduceSum()) * scale
			output[t*conv.outChannels+oc+2] = v2 + conv.bias[oc+2]
			v3 := float32(c3l.Add(c3h).ReduceSum()) * scale
			output[t*conv.outChannels+oc+3] = v3 + conv.bias[oc+3]
		}
	}
	return outputLength
}
