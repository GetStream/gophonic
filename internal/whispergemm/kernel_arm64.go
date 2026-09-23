// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package whispergemm

import "simd/archsimd"

const kernelName = "arm64-neon-4x16"

func mulPacked(dst []float32, dstStride int, a []float32, aStride int, packed []float32, m, k, n int) {
	for c := 0; c < n; c += panelColumns {
		width := min(panelColumns, n-c)
		weights := packed[c*k : (c+panelColumns)*k]
		r := 0
		if width == panelColumns {
			for ; r+4 <= m; r += 4 {
				kernel4x16(
					a[r*aStride:r*aStride+k], a[(r+1)*aStride:(r+1)*aStride+k],
					a[(r+2)*aStride:(r+2)*aStride+k], a[(r+3)*aStride:(r+3)*aStride+k], weights,
					dst[r*dstStride+c:r*dstStride+c+width], dst[(r+1)*dstStride+c:(r+1)*dstStride+c+width],
					dst[(r+2)*dstStride+c:(r+2)*dstStride+c+width], dst[(r+3)*dstStride+c:(r+3)*dstStride+c+width],
				)
			}
		}
		for ; r < m; r++ {
			kernel1x16(a[r*aStride:r*aStride+k], weights, dst[r*dstStride+c:r*dstStride+c+width])
		}
	}
}

// Two K values share each input vector load. Bitwise interleaves broadcast
// each value without scalar GetElem temporaries, keeping the tile in registers.
func kernel4x16(a0, a1, a2, a3, weights, d0, d1, d2, d3 []float32) {
	k := len(a0)
	_ = a1[k-1]
	_ = a2[k-1]
	_ = a3[k-1]
	_ = weights[k*16-1]
	var c00, c01, c02, c03 archsimd.Float32x4
	var c10, c11, c12, c13 archsimd.Float32x4
	var c20, c21, c22, c23 archsimd.Float32x4
	var c30, c31, c32, c33 archsimd.Float32x4
	p := 0
	for ; p+4 <= k; p += 2 {
		v0 := archsimd.LoadFloat32x4Array((*[4]float32)(a0[p : p+4])).ToBits()
		pair0 := v0.InterleaveLo(v0).ReshapeToUint64s()
		v1 := archsimd.LoadFloat32x4Array((*[4]float32)(a1[p : p+4])).ToBits()
		pair1 := v1.InterleaveLo(v1).ReshapeToUint64s()
		v2 := archsimd.LoadFloat32x4Array((*[4]float32)(a2[p : p+4])).ToBits()
		pair2 := v2.InterleaveLo(v2).ReshapeToUint64s()
		v3 := archsimd.LoadFloat32x4Array((*[4]float32)(a3[p : p+4])).ToBits()
		pair3 := v3.InterleaveLo(v3).ReshapeToUint64s()
		wb := (*[32]float32)(weights[p*16:])
		{
			w0 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[0:4]))
			w1 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[4:8]))
			w2 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[8:12]))
			w3 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[12:16]))
			x0 := pair0.InterleaveLo(pair0).ReshapeToUint32s().BitsToFloat32()
			c00 = w0.MulAdd(x0, c00)
			c01 = w1.MulAdd(x0, c01)
			c02 = w2.MulAdd(x0, c02)
			c03 = w3.MulAdd(x0, c03)
			x1 := pair1.InterleaveLo(pair1).ReshapeToUint32s().BitsToFloat32()
			c10 = w0.MulAdd(x1, c10)
			c11 = w1.MulAdd(x1, c11)
			c12 = w2.MulAdd(x1, c12)
			c13 = w3.MulAdd(x1, c13)
			x2 := pair2.InterleaveLo(pair2).ReshapeToUint32s().BitsToFloat32()
			c20 = w0.MulAdd(x2, c20)
			c21 = w1.MulAdd(x2, c21)
			c22 = w2.MulAdd(x2, c22)
			c23 = w3.MulAdd(x2, c23)
			x3 := pair3.InterleaveLo(pair3).ReshapeToUint32s().BitsToFloat32()
			c30 = w0.MulAdd(x3, c30)
			c31 = w1.MulAdd(x3, c31)
			c32 = w2.MulAdd(x3, c32)
			c33 = w3.MulAdd(x3, c33)
		}
		{
			w0 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[16:20]))
			w1 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[20:24]))
			w2 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[24:28]))
			w3 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[28:32]))
			x0 := pair0.InterleaveHi(pair0).ReshapeToUint32s().BitsToFloat32()
			c00 = w0.MulAdd(x0, c00)
			c01 = w1.MulAdd(x0, c01)
			c02 = w2.MulAdd(x0, c02)
			c03 = w3.MulAdd(x0, c03)
			x1 := pair1.InterleaveHi(pair1).ReshapeToUint32s().BitsToFloat32()
			c10 = w0.MulAdd(x1, c10)
			c11 = w1.MulAdd(x1, c11)
			c12 = w2.MulAdd(x1, c12)
			c13 = w3.MulAdd(x1, c13)
			x2 := pair2.InterleaveHi(pair2).ReshapeToUint32s().BitsToFloat32()
			c20 = w0.MulAdd(x2, c20)
			c21 = w1.MulAdd(x2, c21)
			c22 = w2.MulAdd(x2, c22)
			c23 = w3.MulAdd(x2, c23)
			x3 := pair3.InterleaveHi(pair3).ReshapeToUint32s().BitsToFloat32()
			c30 = w0.MulAdd(x3, c30)
			c31 = w1.MulAdd(x3, c31)
			c32 = w2.MulAdd(x3, c32)
			c33 = w3.MulAdd(x3, c33)
		}
	}
	for ; p < k; p++ {
		wb := (*[16]float32)(weights[p*16:])
		w0 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[0:4]))
		w1 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[4:8]))
		w2 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[8:12]))
		w3 := archsimd.LoadFloat32x4Array((*[4]float32)(wb[12:16]))
		x0 := archsimd.BroadcastFloat32x4(a0[p])
		c00 = w0.MulAdd(x0, c00)
		c01 = w1.MulAdd(x0, c01)
		c02 = w2.MulAdd(x0, c02)
		c03 = w3.MulAdd(x0, c03)
		x1 := archsimd.BroadcastFloat32x4(a1[p])
		c10 = w0.MulAdd(x1, c10)
		c11 = w1.MulAdd(x1, c11)
		c12 = w2.MulAdd(x1, c12)
		c13 = w3.MulAdd(x1, c13)
		x2 := archsimd.BroadcastFloat32x4(a2[p])
		c20 = w0.MulAdd(x2, c20)
		c21 = w1.MulAdd(x2, c21)
		c22 = w2.MulAdd(x2, c22)
		c23 = w3.MulAdd(x2, c23)
		x3 := archsimd.BroadcastFloat32x4(a3[p])
		c30 = w0.MulAdd(x3, c30)
		c31 = w1.MulAdd(x3, c31)
		c32 = w2.MulAdd(x3, c32)
		c33 = w3.MulAdd(x3, c33)
	}
	_ = d0[15]
	c00.StoreArray((*[4]float32)(d0[0:4]))
	c01.StoreArray((*[4]float32)(d0[4:8]))
	c02.StoreArray((*[4]float32)(d0[8:12]))
	c03.StoreArray((*[4]float32)(d0[12:16]))
	_ = d1[15]
	c10.StoreArray((*[4]float32)(d1[0:4]))
	c11.StoreArray((*[4]float32)(d1[4:8]))
	c12.StoreArray((*[4]float32)(d1[8:12]))
	c13.StoreArray((*[4]float32)(d1[12:16]))
	_ = d2[15]
	c20.StoreArray((*[4]float32)(d2[0:4]))
	c21.StoreArray((*[4]float32)(d2[4:8]))
	c22.StoreArray((*[4]float32)(d2[8:12]))
	c23.StoreArray((*[4]float32)(d2[12:16]))
	_ = d3[15]
	c30.StoreArray((*[4]float32)(d3[0:4]))
	c31.StoreArray((*[4]float32)(d3[4:8]))
	c32.StoreArray((*[4]float32)(d3[8:12]))
	c33.StoreArray((*[4]float32)(d3[12:16]))
}

// Independent K streams avoid the long FMA dependency chain in single-row
// decoding. The final reduction order is ((stream0+stream1)+stream2)+stream3.
func kernel1x16(a, weights, dst []float32) {
	var c00, c01, c02, c03 archsimd.Float32x4
	var c10, c11, c12, c13 archsimd.Float32x4
	var c20, c21, c22, c23 archsimd.Float32x4
	var c30, c31, c32, c33 archsimd.Float32x4
	k := len(a)
	p := 0
	for ; p+4 <= k; p += 4 {
		v := archsimd.LoadFloat32x4Array((*[4]float32)(a[p : p+4])).ToBits()
		lo := v.InterleaveLo(v).ReshapeToUint64s()
		hi := v.InterleaveHi(v).ReshapeToUint64s()
		wb := (*[64]float32)(weights[p*16:])
		x0 := lo.InterleaveLo(lo).ReshapeToUint32s().BitsToFloat32()
		c00 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[0:4])).MulAdd(x0, c00)
		c01 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[4:8])).MulAdd(x0, c01)
		c02 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[8:12])).MulAdd(x0, c02)
		c03 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[12:16])).MulAdd(x0, c03)
		x1 := lo.InterleaveHi(lo).ReshapeToUint32s().BitsToFloat32()
		c10 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[16:20])).MulAdd(x1, c10)
		c11 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[20:24])).MulAdd(x1, c11)
		c12 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[24:28])).MulAdd(x1, c12)
		c13 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[28:32])).MulAdd(x1, c13)
		x2 := hi.InterleaveLo(hi).ReshapeToUint32s().BitsToFloat32()
		c20 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[32:36])).MulAdd(x2, c20)
		c21 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[36:40])).MulAdd(x2, c21)
		c22 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[40:44])).MulAdd(x2, c22)
		c23 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[44:48])).MulAdd(x2, c23)
		x3 := hi.InterleaveHi(hi).ReshapeToUint32s().BitsToFloat32()
		c30 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[48:52])).MulAdd(x3, c30)
		c31 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[52:56])).MulAdd(x3, c31)
		c32 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[56:60])).MulAdd(x3, c32)
		c33 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[60:64])).MulAdd(x3, c33)
	}
	c00 = c00.Add(c10).Add(c20).Add(c30)
	c01 = c01.Add(c11).Add(c21).Add(c31)
	c02 = c02.Add(c12).Add(c22).Add(c32)
	c03 = c03.Add(c13).Add(c23).Add(c33)
	for ; p < k; p++ {
		wb := (*[16]float32)(weights[p*16:])
		x := archsimd.BroadcastFloat32x4(a[p])
		c00 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[0:4])).MulAdd(x, c00)
		c01 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[4:8])).MulAdd(x, c01)
		c02 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[8:12])).MulAdd(x, c02)
		c03 = archsimd.LoadFloat32x4Array((*[4]float32)(wb[12:16])).MulAdd(x, c03)
	}
	if len(dst) == 16 {
		c00.StoreArray((*[4]float32)(dst[0:4]))
		c01.StoreArray((*[4]float32)(dst[4:8]))
		c02.StoreArray((*[4]float32)(dst[8:12]))
		c03.StoreArray((*[4]float32)(dst[12:16]))
		return
	}
	var tail [16]float32
	c00.StoreArray((*[4]float32)(tail[0:4]))
	c01.StoreArray((*[4]float32)(tail[4:8]))
	c02.StoreArray((*[4]float32)(tail[8:12]))
	c03.StoreArray((*[4]float32)(tail[12:16]))
	copy(dst, tail[:len(dst)])
}
