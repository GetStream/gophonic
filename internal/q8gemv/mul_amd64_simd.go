// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package q8gemv

import "simd/archsimd"

// mulKernel widens signed bytes in registers; weights are read once and no
// float32 copy or per-call scratch is needed. Four independent AVX2/FMA chains
// cover 32 input values per iteration.
func mulKernel(x []float32, q []int8, scales, dst []float32, k, n int) {
	for row := 0; row < n; row++ {
		dst[row] = mulRowAMD64(x, q[row*k:(row+1)*k], scales[row])
	}
}

//go:nosplit
func mulRowAMD64(x []float32, qRow []int8, scale float32) float32 {
	k := len(x)
	var c0, c1, c2, c3 archsimd.Float32x8
	i := 0
	for ; i+32 <= k; i += 32 {
		q0 := archsimd.LoadInt8x16(qRow[i:]).ExtendToInt16()
		q1 := archsimd.LoadInt8x16(qRow[i+16:]).ExtendToInt16()
		w0 := q0.GetLo().ExtendToInt32().ConvertToFloat32()
		w1 := q0.GetHi().ExtendToInt32().ConvertToFloat32()
		w2 := q1.GetLo().ExtendToInt32().ConvertToFloat32()
		w3 := q1.GetHi().ExtendToInt32().ConvertToFloat32()
		c0 = archsimd.LoadFloat32x8(x[i:]).MulAdd(w0, c0)
		c1 = archsimd.LoadFloat32x8(x[i+8:]).MulAdd(w1, c1)
		c2 = archsimd.LoadFloat32x8(x[i+16:]).MulAdd(w2, c2)
		c3 = archsimd.LoadFloat32x8(x[i+24:]).MulAdd(w3, c3)
	}
	v := c0.Add(c1).Add(c2).Add(c3)
	halves := v.GetLo().Add(v.GetHi())
	var lanes [4]float32
	halves.StoreArray(&lanes)
	sum := (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
	for ; i < k; i++ {
		sum += x[i] * float32(qRow[i])
	}
	return sum * scale
}
