// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package gophonic

import (
	"simd/archsimd"
	"unsafe"
)

// runTinyStemTile2x4Range computes two adjacent stem frames and four output
// channels at a time. The stem's stride is two, so adjacent outputs share
// input taps; each packed weight vector is loaded once and contributes to both
// frame accumulators.
//
// The supported model stem has groups=1, kernel=5, stride=2, and dimensions
// divisible by the vector/tile widths. A final unpaired frame and unsupported
// shapes use the existing tile4 kernel.
func runTinyStemTile2x4Range(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength, fromT, toT int) int {
	if conv.groups != 1 || conv.kernel != 5 || conv.stride != 2 || conv.inChannels%16 != 0 || conv.outChannels%4 != 0 || len(conv.packed) != conv.outChannels*conv.kernel*conv.inChannels {
		return runTinyConvTile4Range(conv, input, params, output, inputLength, fromT, toT)
	}
	const pad = 2
	outputLength := (inputLength+2*pad-conv.kernel)/conv.stride + 1
	if fromT < 0 {
		fromT = 0
	}
	if toT > outputLength {
		toT = outputLength
	}
	if fromT >= toT {
		return outputLength
	}

	scale := params.scale * conv.weightScale
	zero16 := archsimd.BroadcastUint16x8(uint16(params.zero))
	wzero16 := archsimd.BroadcastUint16x8(uint16(conv.weightZero))
	inputData := unsafe.Pointer(unsafe.SliceData(input))
	weightData := unsafe.Pointer(unsafe.SliceData(conv.packed))
	weightStride := conv.kernel * conv.inChannels
	t := fromT
	for ; t+1 < toT; t += 2 {
		for oc := 0; oc < conv.outChannels; oc += 4 {
			var r00l, r00h, r01l, r01h archsimd.Int32x4
			var r02l, r02h, r03l, r03h archsimd.Int32x4
			var r10l, r10h, r11l, r11h archsimd.Int32x4
			var r12l, r12h, r13l, r13h archsimd.Int32x4
			for k := 0; k < 5; k++ {
				it0 := t*2 + k - pad
				it1 := it0 + 2
				valid0 := it0 >= 0 && it0 < inputLength
				valid1 := it1 >= 0 && it1 < inputLength
				for ic := 0; ic < conv.inChannels; ic += 16 {
					var x0l, x0h, x1l, x1h archsimd.Int16x8
					if valid0 {
						x0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(inputData, uintptr(it0*conv.inChannels+ic))))
						x0l = x0.ExtendLo8ToUint16().Sub(zero16).BitsToInt16()
						x0h = x0.HiToLo().ExtendLo8ToUint16().Sub(zero16).BitsToInt16()
					}
					if valid1 {
						x1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(inputData, uintptr(it1*conv.inChannels+ic))))
						x1l = x1.ExtendLo8ToUint16().Sub(zero16).BitsToInt16()
						x1h = x1.HiToLo().ExtendLo8ToUint16().Sub(zero16).BitsToInt16()
					}
					wp := unsafe.Add(weightData, uintptr(oc*weightStride+k*conv.inChannels+ic))

					w0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(0*weightStride))))
					w0l := w0.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					w0h := w0.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					if valid0 {
						r00l = r00l.Add(x0l.MulWidenLo(w0l)).Add(x0h.MulWidenLo(w0h))
						r00h = r00h.Add(x0l.HiToLo().MulWidenLo(w0l.HiToLo())).Add(x0h.HiToLo().MulWidenLo(w0h.HiToLo()))
					}
					if valid1 {
						r10l = r10l.Add(x1l.MulWidenLo(w0l)).Add(x1h.MulWidenLo(w0h))
						r10h = r10h.Add(x1l.HiToLo().MulWidenLo(w0l.HiToLo())).Add(x1h.HiToLo().MulWidenLo(w0h.HiToLo()))
					}

					w1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(1*weightStride))))
					w1l := w1.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					w1h := w1.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					if valid0 {
						r01l = r01l.Add(x0l.MulWidenLo(w1l)).Add(x0h.MulWidenLo(w1h))
						r01h = r01h.Add(x0l.HiToLo().MulWidenLo(w1l.HiToLo())).Add(x0h.HiToLo().MulWidenLo(w1h.HiToLo()))
					}
					if valid1 {
						r11l = r11l.Add(x1l.MulWidenLo(w1l)).Add(x1h.MulWidenLo(w1h))
						r11h = r11h.Add(x1l.HiToLo().MulWidenLo(w1l.HiToLo())).Add(x1h.HiToLo().MulWidenLo(w1h.HiToLo()))
					}

					w2 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(2*weightStride))))
					w2l := w2.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					w2h := w2.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					if valid0 {
						r02l = r02l.Add(x0l.MulWidenLo(w2l)).Add(x0h.MulWidenLo(w2h))
						r02h = r02h.Add(x0l.HiToLo().MulWidenLo(w2l.HiToLo())).Add(x0h.HiToLo().MulWidenLo(w2h.HiToLo()))
					}
					if valid1 {
						r12l = r12l.Add(x1l.MulWidenLo(w2l)).Add(x1h.MulWidenLo(w2h))
						r12h = r12h.Add(x1l.HiToLo().MulWidenLo(w2l.HiToLo())).Add(x1h.HiToLo().MulWidenLo(w2h.HiToLo()))
					}

					w3 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(wp, uintptr(3*weightStride))))
					w3l := w3.ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					w3h := w3.HiToLo().ExtendLo8ToUint16().Sub(wzero16).BitsToInt16()
					if valid0 {
						r03l = r03l.Add(x0l.MulWidenLo(w3l)).Add(x0h.MulWidenLo(w3h))
						r03h = r03h.Add(x0l.HiToLo().MulWidenLo(w3l.HiToLo())).Add(x0h.HiToLo().MulWidenLo(w3h.HiToLo()))
					}
					if valid1 {
						r13l = r13l.Add(x1l.MulWidenLo(w3l)).Add(x1h.MulWidenLo(w3h))
						r13h = r13h.Add(x1l.HiToLo().MulWidenLo(w3l.HiToLo())).Add(x1h.HiToLo().MulWidenLo(w3h.HiToLo()))
					}
				}
			}

			base0 := t * conv.outChannels
			base1 := base0 + conv.outChannels
			v00 := float32(r00l.Add(r00h).ReduceSum()) * scale
			v01 := float32(r01l.Add(r01h).ReduceSum()) * scale
			v02 := float32(r02l.Add(r02h).ReduceSum()) * scale
			v03 := float32(r03l.Add(r03h).ReduceSum()) * scale
			output[base0+oc+0] = v00 + conv.bias[oc+0]
			output[base0+oc+1] = v01 + conv.bias[oc+1]
			output[base0+oc+2] = v02 + conv.bias[oc+2]
			output[base0+oc+3] = v03 + conv.bias[oc+3]
			v10 := float32(r10l.Add(r10h).ReduceSum()) * scale
			v11 := float32(r11l.Add(r11h).ReduceSum()) * scale
			v12 := float32(r12l.Add(r12h).ReduceSum()) * scale
			v13 := float32(r13l.Add(r13h).ReduceSum()) * scale
			output[base1+oc+0] = v10 + conv.bias[oc+0]
			output[base1+oc+1] = v11 + conv.bias[oc+1]
			output[base1+oc+2] = v12 + conv.bias[oc+2]
			output[base1+oc+3] = v13 + conv.bias[oc+3]
		}
	}
	if t < toT {
		runTinyConvTile4Range(conv, input, params, output, inputLength, t, toT)
	}
	return outputLength
}
