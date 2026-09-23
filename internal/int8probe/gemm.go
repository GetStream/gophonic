// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

// Package int8probe contains an isolated benchmark for Smart Turn's QDQ
// matrix shape. It is intentionally separate from the inference implementation.
package int8probe

const (
	benchRows = 400
	benchK    = 384
	benchCols = 1536
)

// packWeightsKMajor converts [N,K] ONNX-style weights to [K,N], which lets a
// NEON kernel load a contiguous vector of adjacent output channels per K step.
// It also computes the correction term needed when centering QUInt8 inputs at
// 128 while the ONNX activation zero point is not 128.
func packWeightsKMajor(weights []int8, packed []int8, weightSums []int32, rows, cols int) {
	clear(weightSums)
	for n := 0; n < rows; n++ {
		for k := 0; k < cols; k++ {
			v := weights[n*cols+k]
			packed[k*rows+n] = v
			weightSums[n] += int32(v)
		}
	}
}

// quantizedMatMul computes (A-zpA)*W with per-output float scales, returning
// dequantized float32 values. A is QUInt8 row-major, W is QInt8 in [K,N]
// packed order. The function does not allocate.
func quantizedMatMul(a []uint8, packedW []int8, sumW []int32, aScale float32, aZeroPoint int32, wScale, bias, dst []float32) {
	quantizedMatMulKernel(a, packedW, sumW, aScale, aZeroPoint, wScale, bias, dst, benchRows, benchK, benchCols)
}

// float32MatMul is the same [M,K] x [K,N] operation as the current gofloor
// 4x4 microtile, with output-major weights matching gofloor's stored layout.
func float32MatMul(a, weights, bias, dst []float32) {
	for r := 0; r < benchRows; r += 4 {
		a0 := a[r*benchK : (r+1)*benchK]
		a1 := a[(r+1)*benchK : (r+2)*benchK]
		a2 := a[(r+2)*benchK : (r+3)*benchK]
		a3 := a[(r+3)*benchK : (r+4)*benchK]
		for n := 0; n < benchCols; n += 4 {
			var tile [16]float32
			probeDot4x4(
				a0, a1, a2, a3,
				weights[n*benchK:(n+1)*benchK],
				weights[(n+1)*benchK:(n+2)*benchK],
				weights[(n+2)*benchK:(n+3)*benchK],
				weights[(n+3)*benchK:(n+4)*benchK],
				&tile,
			)
			for i := 0; i < 4; i++ {
				base := (r+i)*benchCols + n
				dst[base] = tile[i*4] + bias[n]
				dst[base+1] = tile[i*4+1] + bias[n+1]
				dst[base+2] = tile[i*4+2] + bias[n+2]
				dst[base+3] = tile[i*4+3] + bias[n+3]
			}
		}
	}
}
