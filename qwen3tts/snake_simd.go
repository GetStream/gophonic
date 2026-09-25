// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package qwen3tts

import "simd/archsimd"

// apply writes SnakeBeta of rows of n channels from src, plus bias when
// given, to dst.
func (s *snake) apply(dst, src, bias []float32, n int) {
	if n%4 != 0 {
		snakeScalar(dst, src, bias, s.a, s.invB, n)
		return
	}
	inv := archsimd.BroadcastFloat32x4(invPi)
	p1, p2, p3 := archsimd.BroadcastFloat32x4(-pi1), archsimd.BroadcastFloat32x4(-pi2), archsimd.BroadcastFloat32x4(-pi3)
	c3, c5, c7 := archsimd.BroadcastFloat32x4(s3), archsimd.BroadcastFloat32x4(s5), archsimd.BroadcastFloat32x4(s7)
	c9, c11 := archsimd.BroadcastFloat32x4(s9), archsimd.BroadcastFloat32x4(s11)
	for r := 0; r < len(src); r += n {
		x, y := src[r:r+n], dst[r:r+n]
		for i := 0; i < n; i += 4 {
			v := archsimd.LoadFloat32x4Array((*[4]float32)(x[i : i+4]))
			if bias != nil {
				v = v.Add(archsimd.LoadFloat32x4Array((*[4]float32)(bias[i : i+4])))
			}
			a := archsimd.LoadFloat32x4Array((*[4]float32)(s.a[i : i+4]))
			b := archsimd.LoadFloat32x4Array((*[4]float32)(s.invB[i : i+4]))
			arg := v.Mul(a)
			k := arg.Mul(inv).Round()
			red := k.MulAdd(p1, arg)
			red = k.MulAdd(p2, red)
			red = k.MulAdd(p3, red)
			r2 := red.Mul(red)
			poly := r2.MulAdd(c11, c9)
			poly = r2.MulAdd(poly, c7)
			poly = r2.MulAdd(poly, c5)
			poly = r2.MulAdd(poly, c3)
			sine := red.Mul(r2).MulAdd(poly, red)
			sq := sine.Mul(sine)
			b.MulAdd(sq, v).StoreArray((*[4]float32)(y[i : i+4]))
		}
	}
}

func addTo(dst, src []float32) {
	src = src[:len(dst)]
	i := 0
	for ; i+4 <= len(dst); i += 4 {
		a := archsimd.LoadFloat32x4Array((*[4]float32)(dst[i : i+4]))
		b := archsimd.LoadFloat32x4Array((*[4]float32)(src[i : i+4]))
		a.Add(b).StoreArray((*[4]float32)(dst[i : i+4]))
	}
	for ; i < len(dst); i++ {
		dst[i] += src[i]
	}
}

// residual writes rows of n channels of src + z + bias to dst.
func residual(dst, src, z, bias []float32, n int) {
	if n%4 != 0 {
		residualScalar(dst, src, z, bias, n)
		return
	}
	for r := 0; r < len(dst); r += n {
		x, y, o := src[r:r+n], z[r:r+n], dst[r:r+n]
		for i := 0; i < n; i += 4 {
			a := archsimd.LoadFloat32x4Array((*[4]float32)(x[i : i+4]))
			b := archsimd.LoadFloat32x4Array((*[4]float32)(y[i : i+4]))
			c := archsimd.LoadFloat32x4Array((*[4]float32)(bias[i : i+4]))
			a.Add(b).Add(c).StoreArray((*[4]float32)(o[i : i+4]))
		}
	}
}
