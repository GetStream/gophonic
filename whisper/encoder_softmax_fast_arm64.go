// Copyright 2026 The gophonic authors
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

//go:build goexperiment.simd && arm64

package whisper

import "simd/archsimd"

// softmaxExpRowFallback replaces x with exp(x-max(x)) and returns 1/sum. Callers
// fold the normalization into a later, narrower product. Inputs are clamped
// at ln(2^-126) after subtracting the maximum, so every scale factor is a
// normal FP32 power of two and no special-case path is needed; clamped terms
// are below 1.2e-38 against a sum of at least one. The polynomial is ggml's
// NEON exponential. Four independent partial sums keep the FMA pipes busy.
func softmaxExpRowFallback(x []float32) float32 {
	n := len(x)
	i := 0
	var maxValue float32
	if n >= 16 {
		m0 := archsimd.LoadFloat32x4Array((*[4]float32)(x[0:4]))
		m1 := archsimd.LoadFloat32x4Array((*[4]float32)(x[4:8]))
		m2 := archsimd.LoadFloat32x4Array((*[4]float32)(x[8:12]))
		m3 := archsimd.LoadFloat32x4Array((*[4]float32)(x[12:16]))
		for i = 16; i+16 <= n; i += 16 {
			v := (*[16]float32)(x[i : i+16])
			m0 = m0.Max(archsimd.LoadFloat32x4Array((*[4]float32)(v[0:4])))
			m1 = m1.Max(archsimd.LoadFloat32x4Array((*[4]float32)(v[4:8])))
			m2 = m2.Max(archsimd.LoadFloat32x4Array((*[4]float32)(v[8:12])))
			m3 = m3.Max(archsimd.LoadFloat32x4Array((*[4]float32)(v[12:16])))
		}
		maxValue = m0.Max(m1).Max(m2.Max(m3)).ReduceMax()
	} else {
		maxValue = x[0]
	}
	for ; i < n; i++ {
		maxValue = max(maxValue, x[i])
	}
	maximum := archsimd.BroadcastFloat32x4(maxValue)
	floor := archsimd.BroadcastFloat32x4(-87.33654)
	magic := archsimd.BroadcastFloat32x4(0x1.8p23)
	log2e := archsimd.BroadcastFloat32x4(0x1.715476p+0)
	negLn2High := archsimd.BroadcastFloat32x4(-0x1.62e4p-1)
	negLn2Low := archsimd.BroadcastFloat32x4(-0x1.7f7d1cp-20)
	p0 := archsimd.BroadcastFloat32x4(0x1.ffffecp-1)
	p1 := archsimd.BroadcastFloat32x4(0x1.fffdb6p-2)
	p2 := archsimd.BroadcastFloat32x4(0x1.555e66p-3)
	p3 := archsimd.BroadcastFloat32x4(0x1.573e2ep-5)
	p4 := archsimd.BroadcastFloat32x4(0x1.0e4020p-7)
	oneBits := archsimd.BroadcastUint32x4(0x3f800000)
	exp := func(v archsimd.Float32x4) archsimd.Float32x4 {
		input := v.Sub(maximum).Max(floor)
		z := input.MulAdd(log2e, magic)
		exponent := z.Sub(magic)
		b := exponent.MulAdd(negLn2High, input)
		b = exponent.MulAdd(negLn2Low, b)
		k := z.ToBits().ShiftAllLeft(23).Add(oneBits).BitsToFloat32()
		u := b.Mul(b)
		j := p4.MulAdd(b, p3).MulAdd(u, p2.MulAdd(b, p1)).MulAdd(u, p0.Mul(b))
		return j.MulAdd(k, k)
	}
	var s0, s1, s2, s3 archsimd.Float32x4
	i = 0
	for ; i+16 <= n; i += 16 {
		v := (*[16]float32)(x[i : i+16])
		e0 := exp(archsimd.LoadFloat32x4Array((*[4]float32)(v[0:4])))
		e1 := exp(archsimd.LoadFloat32x4Array((*[4]float32)(v[4:8])))
		e2 := exp(archsimd.LoadFloat32x4Array((*[4]float32)(v[8:12])))
		e3 := exp(archsimd.LoadFloat32x4Array((*[4]float32)(v[12:16])))
		e0.StoreArray((*[4]float32)(v[0:4]))
		e1.StoreArray((*[4]float32)(v[4:8]))
		e2.StoreArray((*[4]float32)(v[8:12]))
		e3.StoreArray((*[4]float32)(v[12:16]))
		s0, s1, s2, s3 = s0.Add(e0), s1.Add(e1), s2.Add(e2), s3.Add(e3)
	}
	for ; i+4 <= n; i += 4 {
		e := exp(archsimd.LoadFloat32x4Array((*[4]float32)(x[i : i+4])))
		e.StoreArray((*[4]float32)(x[i : i+4]))
		s0 = s0.Add(e)
	}
	sum := s0.Add(s1).Add(s2.Add(s3))
	total := (sum.GetElem(0) + sum.GetElem(1)) + (sum.GetElem(2) + sum.GetElem(3))
	if i < n {
		var tail [4]float32
		for j := range tail {
			tail[j] = -87.33654 + maxValue - 1 // below the floor after subtraction
		}
		copy(tail[:], x[i:])
		e := exp(archsimd.LoadFloat32x4Array(&tail))
		e.StoreArray(&tail)
		for j := i; j < n; j++ {
			x[j] = tail[j-i]
			total += tail[j-i]
		}
	}
	return 1 / total
}
