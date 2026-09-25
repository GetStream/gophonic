// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package tinymel

import (
	"simd/archsimd"
	"unsafe"
)

// runTinyConvSIMD dispatches the full output-time range to the SIMD-optimized
// kernels, falling back to the original vector reduction implementation when
// a layer shape is not supported by a specialized kernel.
func runTinyConvSIMD(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength int) int {
	pad := conv.kernel / 2
	outputLength := (inputLength+2*pad-conv.kernel)/conv.stride + 1
	return runTinyConvRange(conv, input, params, output, inputLength, 0, outputLength)
}

// runTinyConvSIMDRange writes the half-open output time interval [fromT,toT).
// It keeps absolute output indexes when selecting source frames, which lets a
// caller split the time dimension among persistent workers without copies.
func runTinyConvSIMDRange(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength, fromT, toT int) int {
	pad := conv.kernel / 2
	outputLength := (inputLength+2*pad-conv.kernel)/conv.stride + 1
	if fromT < 0 {
		fromT = 0
	}
	if toT > outputLength {
		toT = outputLength
	}
	if conv.groups == conv.inChannels || len(conv.packed) != conv.outChannels*conv.kernel*(conv.inChannels/conv.groups) {
		return runTinyConvScalarRange(conv, input, params, output, inputLength, fromT, toT)
	}
	inPerGroup := conv.inChannels / conv.groups
	outPerGroup := conv.outChannels / conv.groups
	scale := params.scale * conv.weightScale
	zero16 := archsimd.BroadcastUint16x8(uint16(params.zero))
	wzero16 := archsimd.BroadcastUint16x8(uint16(conv.weightZero))
	inputData := unsafe.Pointer(unsafe.SliceData(input))
	weightData := unsafe.Pointer(unsafe.SliceData(conv.packed))

	for t := fromT; t < toT; t++ {
		for oc := 0; oc < conv.outChannels; oc++ {
			inputGroupBase := 0
			if conv.groups != 1 {
				inputGroupBase = (oc / outPerGroup) * inPerGroup
			}
			weightOutputBase := oc * conv.kernel * inPerGroup
			var acc0, acc1, acc2, acc3 archsimd.Int32x4
			var scalarTail int32
			for k := 0; k < conv.kernel; k++ {
				it := t*conv.stride + k - pad
				if it < 0 || it >= inputLength {
					continue
				}
				inputBase := it*conv.inChannels + inputGroupBase
				weightBase := weightOutputBase + k*inPerGroup
				ip := unsafe.Add(inputData, uintptr(inputBase))
				wp := unsafe.Add(weightData, uintptr(weightBase))
				ic := 0
				for ; ic+16 <= inPerGroup; ic += 16 {
					x := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(ip, uintptr(ic))))
					w := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(ic))))
					xLo := x.ExtendLo8ToUint16().Sub(zero16).BitsToInt16()
					xHi := x.HiToLo().ExtendLo8ToUint16().Sub(zero16).BitsToInt16()
					wLo := w.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					wHi := w.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					acc0 = acc0.Add(xLo.MulWidenLo(wLo))
					acc1 = acc1.Add(xLo.HiToLo().MulWidenLo(wLo.HiToLo()))
					acc2 = acc2.Add(xHi.MulWidenLo(wHi))
					acc3 = acc3.Add(xHi.HiToLo().MulWidenLo(wHi.HiToLo()))
				}
				for ; ic < inPerGroup; ic++ {
					x := int32(*(*uint8)(unsafe.Add(ip, uintptr(ic)))) - int32(params.zero)
					w := int32(*(*uint8)(unsafe.Add(wp, uintptr(ic)))) - int32(conv.weightZero)
					scalarTail += x * w
				}
			}
			acc := simdTinyConvReduce(acc0) + simdTinyConvReduce(acc1) + simdTinyConvReduce(acc2) + simdTinyConvReduce(acc3) + scalarTail
			value := float32(acc) * scale
			output[t*conv.outChannels+oc] = value + conv.bias[oc]
		}
	}
	return outputLength
}

// simdTinyConvReduce returns the sum of the four int32 lanes. It is kept
// out of the hot loop; the vector reductions themselves stay register based.
func simdTinyConvReduce(value archsimd.Int32x4) int32 {
	return value.ReduceSum()
}
