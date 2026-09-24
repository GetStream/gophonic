// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whispergemm

import "simd/archsimd"

const kernelName = "amd64-avx-2x16-1x16"

// The packed panels already put sixteen output columns next to each other.
// Two rows reuse each weight load while keeping four accumulators, two weight
// vectors, and two broadcasts within amd64's sixteen vector registers.
func mulPacked(dst []float32, dstStride int, a []float32, aStride int, packed []float32, m, k, n int) {
	for c := 0; c < n; c += panelColumns {
		width := min(panelColumns, n-c)
		weights := packed[c*k : (c+panelColumns)*k]
		r := 0
		if width == panelColumns {
			for ; r+2 <= m; r += 2 {
				kernel2x16AMD64(
					a[r*aStride:r*aStride+k], a[(r+1)*aStride:(r+1)*aStride+k], weights,
					dst[r*dstStride+c:r*dstStride+c+panelColumns],
					dst[(r+1)*dstStride+c:(r+1)*dstStride+c+panelColumns],
				)
			}
		}
		for ; r < m; r++ {
			kernel1x16AMD64(a[r*aStride:r*aStride+k], weights, dst[r*dstStride+c:r*dstStride+c+width])
		}
	}
}

//go:nosplit
func kernel2x16AMD64(a0, a1, weights, d0, d1 []float32) {
	var c00, c01, c10, c11 archsimd.Float32x8
	for p := range a0 {
		w := weights[p*panelColumns:]
		lo := archsimd.LoadFloat32x8(w)
		hi := archsimd.LoadFloat32x8(w[8:])
		x0 := archsimd.BroadcastFloat32x8(a0[p])
		c00 = lo.MulAdd(x0, c00)
		c01 = hi.MulAdd(x0, c01)
		x1 := archsimd.BroadcastFloat32x8(a1[p])
		c10 = lo.MulAdd(x1, c10)
		c11 = hi.MulAdd(x1, c11)
	}
	c00.StoreArray((*[8]float32)(d0[:8]))
	c01.StoreArray((*[8]float32)(d0[8:16]))
	c10.StoreArray((*[8]float32)(d1[:8]))
	c11.StoreArray((*[8]float32)(d1[8:16]))
}

// Four independent K streams hide the FMA dependency for one-row decoding.
// Only a partial final panel uses the stack result; full panels write directly.
//
//go:nosplit
func kernel1x16AMD64(a, weights, dst []float32) {
	var c00, c01, c10, c11, c20, c21, c30, c31 archsimd.Float32x8
	p := 0
	for ; p+4 <= len(a); p += 4 {
		w := weights[p*panelColumns:]
		lo := archsimd.LoadFloat32x8(w)
		hi := archsimd.LoadFloat32x8(w[8:])
		x := archsimd.BroadcastFloat32x8(a[p])
		c00 = lo.MulAdd(x, c00)
		c01 = hi.MulAdd(x, c01)
		lo = archsimd.LoadFloat32x8(w[16:])
		hi = archsimd.LoadFloat32x8(w[24:])
		x = archsimd.BroadcastFloat32x8(a[p+1])
		c10 = lo.MulAdd(x, c10)
		c11 = hi.MulAdd(x, c11)
		lo = archsimd.LoadFloat32x8(w[32:])
		hi = archsimd.LoadFloat32x8(w[40:])
		x = archsimd.BroadcastFloat32x8(a[p+2])
		c20 = lo.MulAdd(x, c20)
		c21 = hi.MulAdd(x, c21)
		lo = archsimd.LoadFloat32x8(w[48:])
		hi = archsimd.LoadFloat32x8(w[56:])
		x = archsimd.BroadcastFloat32x8(a[p+3])
		c30 = lo.MulAdd(x, c30)
		c31 = hi.MulAdd(x, c31)
	}
	c00 = c00.Add(c10).Add(c20).Add(c30)
	c01 = c01.Add(c11).Add(c21).Add(c31)
	for ; p < len(a); p++ {
		w := weights[p*panelColumns:]
		x := archsimd.BroadcastFloat32x8(a[p])
		c00 = archsimd.LoadFloat32x8(w).MulAdd(x, c00)
		c01 = archsimd.LoadFloat32x8(w[8:]).MulAdd(x, c01)
	}
	if len(dst) == panelColumns {
		c00.StoreArray((*[8]float32)(dst[:8]))
		c01.StoreArray((*[8]float32)(dst[8:16]))
		return
	}
	var tail [panelColumns]float32
	c00.StoreArray((*[8]float32)(tail[:8]))
	c01.StoreArray((*[8]float32)(tail[8:]))
	copy(dst, tail[:])
}
