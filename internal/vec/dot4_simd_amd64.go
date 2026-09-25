// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64

package vec

import "simd/archsimd"

// Dot4 is the amd64 counterpart of dot4_simd.go. It keeps the same
// 16-elements-per-iteration stride but uses Float32x8 with a 2-way K unroll:
// 8 live accumulators plus 2 sources instead of 16 accumulators plus 4
// sources, which no longer spills vector registers on AVX2.
func Dot4(a, b0, b1, b2, b3 []float32) (float32, float32, float32, float32) {
	n := len(a)
	var c00, c01 archsimd.Float32x8
	var c10, c11 archsimd.Float32x8
	var c20, c21 archsimd.Float32x8
	var c30, c31 archsimd.Float32x8
	i := 0
	for ; i+16 <= n; i += 16 {
		a0 := archsimd.LoadFloat32x8(a[i:])
		a1 := archsimd.LoadFloat32x8(a[i+8:])
		c00 = a0.MulAdd(archsimd.LoadFloat32x8(b0[i:]), c00)
		c01 = a1.MulAdd(archsimd.LoadFloat32x8(b0[i+8:]), c01)
		c10 = a0.MulAdd(archsimd.LoadFloat32x8(b1[i:]), c10)
		c11 = a1.MulAdd(archsimd.LoadFloat32x8(b1[i+8:]), c11)
		c20 = a0.MulAdd(archsimd.LoadFloat32x8(b2[i:]), c20)
		c21 = a1.MulAdd(archsimd.LoadFloat32x8(b2[i+8:]), c21)
		c30 = a0.MulAdd(archsimd.LoadFloat32x8(b3[i:]), c30)
		c31 = a1.MulAdd(archsimd.LoadFloat32x8(b3[i+8:]), c31)
	}
	s0 := reducePair8(c00, c01)
	s1 := reducePair8(c10, c11)
	s2 := reducePair8(c20, c21)
	s3 := reducePair8(c30, c31)
	for ; i < n; i++ {
		x := a[i]
		s0 += x * b0[i]
		s1 += x * b1[i]
		s2 += x * b2[i]
		s3 += x * b3[i]
	}
	return s0, s1, s2, s3
}

func reducePair8(x, y archsimd.Float32x8) float32 {
	return reduceVector8(x.Add(y))
}
