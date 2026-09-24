// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package int8probe

import (
	"simd/archsimd"
	"unsafe"
)

func quantizedMatMulKernel(a []uint8, packedW []int8, sumW []int32, aScale float32, aZeroPoint int32, wScale, bias, dst []float32, rows, k, cols int) {
	correction := int32(128) - aZeroPoint
	activationScale := aScale
	for r := 0; r < rows; r++ {
		aRow := a[r*k : (r+1)*k]
		outRow := dst[r*cols : (r+1)*cols]
		for n := 0; n < cols; n += 16 {
			var acc0, acc1, acc2, acc3 archsimd.Int32x4
			for p := 0; p+1 < k; p += 2 {
				a0 := archsimd.BroadcastInt8x16(int8(int(aRow[p]) - 128))
				a1 := archsimd.BroadcastInt8x16(int8(int(aRow[p+1]) - 128))
				w0 := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&packedW[p*cols+n])))
				w1 := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&packedW[(p+1)*cols+n])))
				lo := a0.MulWidenLo(w0).Add(a1.MulWidenLo(w1))
				hi := a0.HiToLo().MulWidenLo(w0.HiToLo()).Add(a1.HiToLo().MulWidenLo(w1.HiToLo()))
				acc0 = acc0.Add(lo.ExtendLo4ToInt32())
				acc1 = acc1.Add(lo.HiToLo().ExtendLo4ToInt32())
				acc2 = acc2.Add(hi.ExtendLo4ToInt32())
				acc3 = acc3.Add(hi.HiToLo().ExtendLo4ToInt32())
			}
			var s0, s1, s2, s3 [4]int32
			acc0.StoreArray(&s0)
			acc1.StoreArray(&s1)
			acc2.StoreArray(&s2)
			acc3.StoreArray(&s3)
			for lane := 0; lane < 4; lane++ {
				index := n + lane
				dot0, dot1, dot2, dot3 := s0[lane], s1[lane], s2[lane], s3[lane]
				if k&1 != 0 {
					p := k - 1
					centered := int32(int(aRow[p]) - 128)
					dot0 += centered * int32(packedW[p*cols+index])
					dot1 += centered * int32(packedW[p*cols+index+4])
					dot2 += centered * int32(packedW[p*cols+index+8])
					dot3 += centered * int32(packedW[p*cols+index+12])
				}
				outRow[index] = float32(dot0+correction*sumW[index])*activationScale*wScale[index] + bias[index]
				index += 4
				outRow[index] = float32(dot1+correction*sumW[index])*activationScale*wScale[index] + bias[index]
				index += 4
				outRow[index] = float32(dot2+correction*sumW[index])*activationScale*wScale[index] + bias[index]
				index += 4
				outRow[index] = float32(dot3+correction*sumW[index])*activationScale*wScale[index] + bias[index]
			}
		}
	}
}
