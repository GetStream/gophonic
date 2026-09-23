// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package int8probe

func quantizedMatMulKernel(a []uint8, packedW []int8, sumW []int32, aScale float32, aZeroPoint int32, wScale, bias, dst []float32, rows, k, cols int) {
	correction := int32(128) - aZeroPoint
	for r := 0; r < rows; r++ {
		aRow := a[r*k : (r+1)*k]
		outRow := dst[r*cols : (r+1)*cols]
		for n := 0; n < cols; n++ {
			var sum int32
			for p, aq := range aRow {
				sum += int32(int8(int(aq)-128)) * int32(packedW[p*cols+n])
			}
			sum += correction * sumW[n]
			outRow[n] = float32(sum)*aScale*wScale[n] + bias[n]
		}
	}
}
