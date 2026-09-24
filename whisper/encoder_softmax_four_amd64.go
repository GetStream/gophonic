// Copyright 2026 The gophonic authors
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//
// The exponential polynomial and special-case handling match
// encoder_softmax_amd64.go and its independent pinned ggml NEON oracle.

//go:build goexperiment.simd && amd64 && amd64.v3

package whisper

import (
	"math"
	"simd/archsimd"
)

// softmaxFourRows keeps independent queries in SIMD lanes when adding the
// exponential terms. Each lane still adds columns in increasing order, exactly
// like softmaxRow; no reduction tree or partial sum changes its rounding.
// Four-by-four register transposes retain the existing row-major score layout.
func softmaxFourRows(x0, x1, x2, x3 []float32) {
	n := len(x0)
	if n < 4 {
		softmaxRow(x0)
		softmaxRow(x1)
		softmaxRow(x2)
		softmaxRow(x3)
		return
	}
	x1, x2, x3 = x1[:n], x2[:n], x3[:n]
	m0 := archsimd.LoadFloat32x4Array((*[4]float32)(x0[:4]))
	m1 := archsimd.LoadFloat32x4Array((*[4]float32)(x1[:4]))
	m2 := archsimd.LoadFloat32x4Array((*[4]float32)(x2[:4]))
	m3 := archsimd.LoadFloat32x4Array((*[4]float32)(x3[:4]))
	i := 4
	for ; i+4 <= n; i += 4 {
		m0 = m0.Max(archsimd.LoadFloat32x4Array((*[4]float32)(x0[i : i+4])))
		m1 = m1.Max(archsimd.LoadFloat32x4Array((*[4]float32)(x1[i : i+4])))
		m2 = m2.Max(archsimd.LoadFloat32x4Array((*[4]float32)(x2[i : i+4])))
		m3 = m3.Max(archsimd.LoadFloat32x4Array((*[4]float32)(x3[i : i+4])))
	}
	max0, max1, max2, max3 := maxFloat4(m0), maxFloat4(m1), maxFloat4(m2), maxFloat4(m3)
	for ; i < n; i++ {
		max0 = max(max0, x0[i])
		max1 = max(max1, x1[i])
		max2 = max(max2, x2[i])
		max3 = max(max3, x3[i])
	}
	maximum := archsimd.Float32x4{}.SetElem(0, max0).SetElem(1, max1).SetElem(2, max2).SetElem(3, max3)
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
	normalLimit := archsimd.BroadcastFloat32x4(126)
	var total archsimd.Float32x4
	i = 0
	for ; i+4 <= n; i += 4 {
		r0 := archsimd.LoadFloat32x4Array((*[4]float32)(x0[i : i+4])).ToBits()
		r1 := archsimd.LoadFloat32x4Array((*[4]float32)(x1[i : i+4])).ToBits()
		r2 := archsimd.LoadFloat32x4Array((*[4]float32)(x2[i : i+4])).ToBits()
		r3 := archsimd.LoadFloat32x4Array((*[4]float32)(x3[i : i+4])).ToBits()
		lo01, hi01 := r0.InterleaveLo(r1).ReshapeToUint64s(), r0.InterleaveHi(r1).ReshapeToUint64s()
		lo23, hi23 := r2.InterleaveLo(r3).ReshapeToUint64s(), r2.InterleaveHi(r3).ReshapeToUint64s()
		v0 := lo01.InterleaveLo(lo23).ReshapeToUint32s().BitsToFloat32()
		v1 := lo01.InterleaveHi(lo23).ReshapeToUint32s().BitsToFloat32()
		v2 := hi01.InterleaveLo(hi23).ReshapeToUint32s().BitsToFloat32()
		v3 := hi01.InterleaveHi(hi23).ReshapeToUint32s().BitsToFloat32()

		{
			input := v0.Sub(maximum)
			z := input.MulAdd(log2e, magic)
			exponent := z.Sub(magic)
			b := exponent.MulAdd(negLn2High, input)
			b = exponent.MulAdd(negLn2Low, b)
			e := z.ToBits().ShiftAllLeft(23)
			k := e.Add(oneBits).BitsToFloat32()
			u := b.Mul(b)
			lower := p2.MulAdd(b, p1)
			upper := p4.MulAdd(b, p3)
			j := upper.MulAdd(u, lower).MulAdd(u, p0.Mul(b))
			v0 = j.MulAdd(k, k)
			special := exponent.Abs().Greater(normalLimit)
			if anyMask4(special.ToInt32x4()) {
				d := archsimd.BroadcastUint32x4(0x82000000).Masked(exponent.LessEqual(archsimd.Float32x4{}))
				s1 := d.Add(archsimd.BroadcastUint32x4(0x7f000000)).BitsToFloat32()
				s2 := e.Sub(d).BitsToFloat32()
				v0 = j.MulAdd(s2, s2).Mul(s1).IfElse(special, v0)
				v0 = s1.Mul(s1).IfElse(exponent.Abs().Greater(archsimd.BroadcastFloat32x4(192)), v0)
			}
			total = total.Add(v0)
		}

		{
			input := v1.Sub(maximum)
			z := input.MulAdd(log2e, magic)
			exponent := z.Sub(magic)
			b := exponent.MulAdd(negLn2High, input)
			b = exponent.MulAdd(negLn2Low, b)
			e := z.ToBits().ShiftAllLeft(23)
			k := e.Add(oneBits).BitsToFloat32()
			u := b.Mul(b)
			lower := p2.MulAdd(b, p1)
			upper := p4.MulAdd(b, p3)
			j := upper.MulAdd(u, lower).MulAdd(u, p0.Mul(b))
			v1 = j.MulAdd(k, k)
			special := exponent.Abs().Greater(normalLimit)
			if anyMask4(special.ToInt32x4()) {
				d := archsimd.BroadcastUint32x4(0x82000000).Masked(exponent.LessEqual(archsimd.Float32x4{}))
				s1 := d.Add(archsimd.BroadcastUint32x4(0x7f000000)).BitsToFloat32()
				s2 := e.Sub(d).BitsToFloat32()
				v1 = j.MulAdd(s2, s2).Mul(s1).IfElse(special, v1)
				v1 = s1.Mul(s1).IfElse(exponent.Abs().Greater(archsimd.BroadcastFloat32x4(192)), v1)
			}
			total = total.Add(v1)
		}

		{
			input := v2.Sub(maximum)
			z := input.MulAdd(log2e, magic)
			exponent := z.Sub(magic)
			b := exponent.MulAdd(negLn2High, input)
			b = exponent.MulAdd(negLn2Low, b)
			e := z.ToBits().ShiftAllLeft(23)
			k := e.Add(oneBits).BitsToFloat32()
			u := b.Mul(b)
			lower := p2.MulAdd(b, p1)
			upper := p4.MulAdd(b, p3)
			j := upper.MulAdd(u, lower).MulAdd(u, p0.Mul(b))
			v2 = j.MulAdd(k, k)
			special := exponent.Abs().Greater(normalLimit)
			if anyMask4(special.ToInt32x4()) {
				d := archsimd.BroadcastUint32x4(0x82000000).Masked(exponent.LessEqual(archsimd.Float32x4{}))
				s1 := d.Add(archsimd.BroadcastUint32x4(0x7f000000)).BitsToFloat32()
				s2 := e.Sub(d).BitsToFloat32()
				v2 = j.MulAdd(s2, s2).Mul(s1).IfElse(special, v2)
				v2 = s1.Mul(s1).IfElse(exponent.Abs().Greater(archsimd.BroadcastFloat32x4(192)), v2)
			}
			total = total.Add(v2)
		}

		{
			input := v3.Sub(maximum)
			z := input.MulAdd(log2e, magic)
			exponent := z.Sub(magic)
			b := exponent.MulAdd(negLn2High, input)
			b = exponent.MulAdd(negLn2Low, b)
			e := z.ToBits().ShiftAllLeft(23)
			k := e.Add(oneBits).BitsToFloat32()
			u := b.Mul(b)
			lower := p2.MulAdd(b, p1)
			upper := p4.MulAdd(b, p3)
			j := upper.MulAdd(u, lower).MulAdd(u, p0.Mul(b))
			v3 = j.MulAdd(k, k)
			special := exponent.Abs().Greater(normalLimit)
			if anyMask4(special.ToInt32x4()) {
				d := archsimd.BroadcastUint32x4(0x82000000).Masked(exponent.LessEqual(archsimd.Float32x4{}))
				s1 := d.Add(archsimd.BroadcastUint32x4(0x7f000000)).BitsToFloat32()
				s2 := e.Sub(d).BitsToFloat32()
				v3 = j.MulAdd(s2, s2).Mul(s1).IfElse(special, v3)
				v3 = s1.Mul(s1).IfElse(exponent.Abs().Greater(archsimd.BroadcastFloat32x4(192)), v3)
			}
			total = total.Add(v3)
		}

		lo01, hi01 = v0.ToBits().InterleaveLo(v1.ToBits()).ReshapeToUint64s(), v0.ToBits().InterleaveHi(v1.ToBits()).ReshapeToUint64s()
		lo23, hi23 = v2.ToBits().InterleaveLo(v3.ToBits()).ReshapeToUint64s(), v2.ToBits().InterleaveHi(v3.ToBits()).ReshapeToUint64s()
		lo01.InterleaveLo(lo23).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(x0[i : i+4]))
		lo01.InterleaveHi(lo23).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(x1[i : i+4]))
		hi01.InterleaveLo(hi23).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(x2[i : i+4]))
		hi01.InterleaveHi(hi23).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(x3[i : i+4]))
	}
	for ; i < n; i++ {
		x0[i] = float32(math.Exp(float64(x0[i] - max0)))
		x1[i] = float32(math.Exp(float64(x1[i] - max1)))
		x2[i] = float32(math.Exp(float64(x2[i] - max2)))
		x3[i] = float32(math.Exp(float64(x3[i] - max3)))
		total = total.Add(archsimd.Float32x4{}.SetElem(0, x0[i]).SetElem(1, x1[i]).SetElem(2, x2[i]).SetElem(3, x3[i]))
	}
	inverse := archsimd.BroadcastFloat32x4(1).Div(total)
	inverse0, inverse1, inverse2, inverse3 := inverse.GetElem(0), inverse.GetElem(1), inverse.GetElem(2), inverse.GetElem(3)
	s0 := archsimd.BroadcastFloat32x4(inverse0)
	s1 := archsimd.BroadcastFloat32x4(inverse1)
	s2 := archsimd.BroadcastFloat32x4(inverse2)
	s3 := archsimd.BroadcastFloat32x4(inverse3)
	i = 0
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat32x4Array((*[4]float32)(x0[i : i+4])).Mul(s0).StoreArray((*[4]float32)(x0[i : i+4]))
		archsimd.LoadFloat32x4Array((*[4]float32)(x1[i : i+4])).Mul(s1).StoreArray((*[4]float32)(x1[i : i+4]))
		archsimd.LoadFloat32x4Array((*[4]float32)(x2[i : i+4])).Mul(s2).StoreArray((*[4]float32)(x2[i : i+4]))
		archsimd.LoadFloat32x4Array((*[4]float32)(x3[i : i+4])).Mul(s3).StoreArray((*[4]float32)(x3[i : i+4]))
	}
	for ; i < n; i++ {
		x0[i] *= inverse0
		x1[i] *= inverse1
		x2[i] *= inverse2
		x3[i] *= inverse3
	}
}
