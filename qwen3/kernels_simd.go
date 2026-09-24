// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package qwen3

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

func load4(s []float32, i int) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&s[i])))
}

func store4(v archsimd.Float32x4, s []float32, i int) {
	v.StoreArray((*[4]float32)(unsafe.Pointer(&s[i])))
}

func sum4(v archsimd.Float32x4) float32 {
	return (v.GetElem(0) + v.GetElem(1)) + (v.GetElem(2) + v.GetElem(3))
}

// exp4 is the vector form of expNonPositive32 for x <= 0.
func exp4(x archsimd.Float32x4) archsimd.Float32x4 {
	x = x.Max(archsimd.BroadcastFloat32x4(-87))
	n := x.Mul(archsimd.BroadcastFloat32x4(1.4426950408889634)).Trunc()
	r := n.MulAdd(archsimd.BroadcastFloat32x4(-ln2Hi), x)
	r = n.MulAdd(archsimd.BroadcastFloat32x4(-ln2Lo), r)
	p := archsimd.BroadcastFloat32x4(1.0 / 40320.0)
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1.0/5040.0))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1.0/720.0))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1.0/120.0))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1.0/24.0))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1.0/6.0))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(0.5))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1))
	// arm64 archsimd has no integer-to-float bit reinterpretation; route the
	// exponent bits through a non-escaping stack buffer instead.
	var bits [4]int32
	n.ConvertToInt32().Add(archsimd.BroadcastInt32x4(127)).ShiftAllLeft(23).StoreArray(&bits)
	scale := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bits)))
	return scale.Mul(p)
}

// swigluInto writes silu(gate)*up into gate.
func swigluInto(gate, up []float32) {
	up = up[:len(gate)]
	zero, one := archsimd.BroadcastFloat32x4(0), archsimd.BroadcastFloat32x4(1)
	i := 0
	for ; i+4 <= len(gate); i += 4 {
		x := load4(gate, i)
		e := exp4(x.Abs().Neg())
		// x >= 0: x/(1+e^-x); x < 0: x*e^x/(1+e^x). Both use e = exp(-|x|).
		num := x.Mul(e).IfElse(x.Less(zero), x)
		store4(num.Div(one.Add(e)).Mul(load4(up, i)), gate, i)
	}
	for ; i < len(gate); i++ {
		gate[i] = silu32(gate[i]) * up[i]
	}
}

func sumSquares(x []float32) float32 {
	var a0, a1, a2, a3 archsimd.Float32x4
	i := 0
	for ; i+16 <= len(x); i += 16 {
		v0, v1, v2, v3 := load4(x, i), load4(x, i+4), load4(x, i+8), load4(x, i+12)
		a0 = v0.MulAdd(v0, a0)
		a1 = v1.MulAdd(v1, a1)
		a2 = v2.MulAdd(v2, a2)
		a3 = v3.MulAdd(v3, a3)
	}
	s := sum4(a0.Add(a1).Add(a2.Add(a3)))
	for ; i < len(x); i++ {
		s += x[i] * x[i]
	}
	return s
}

// scaleMulInto writes src[i]*inv*weight[i] into dst.
func scaleMulInto(dst, src, weight []float32, inv float32) {
	src, weight = src[:len(dst)], weight[:len(dst)]
	s := archsimd.BroadcastFloat32x4(inv)
	i := 0
	for ; i+4 <= len(dst); i += 4 {
		store4(load4(src, i).Mul(s).Mul(load4(weight, i)), dst, i)
	}
	for ; i < len(dst); i++ {
		dst[i] = src[i] * inv * weight[i]
	}
}

func addInto(dst, src []float32) {
	src = src[:len(dst)]
	i := 0
	for ; i+4 <= len(dst); i += 4 {
		store4(load4(dst, i).Add(load4(src, i)), dst, i)
	}
	for ; i < len(dst); i++ {
		dst[i] += src[i]
	}
}

func dot32(a, b []float32) float32 {
	b = b[:len(a)]
	var a0, a1, a2, a3 archsimd.Float32x4
	i := 0
	for ; i+16 <= len(a); i += 16 {
		a0 = load4(a, i).MulAdd(load4(b, i), a0)
		a1 = load4(a, i+4).MulAdd(load4(b, i+4), a1)
		a2 = load4(a, i+8).MulAdd(load4(b, i+8), a2)
		a3 = load4(a, i+12).MulAdd(load4(b, i+12), a3)
	}
	s := sum4(a0.Add(a1).Add(a2.Add(a3)))
	for ; i < len(a); i++ {
		s += a[i] * b[i]
	}
	return s
}

func axpy32(dst, x []float32, a float32) {
	x = x[:len(dst)]
	s := archsimd.BroadcastFloat32x4(a)
	i := 0
	for ; i+4 <= len(dst); i += 4 {
		store4(load4(x, i).MulAdd(s, load4(dst, i)), dst, i)
	}
	for ; i < len(dst); i++ {
		dst[i] += a * x[i]
	}
}

func scaleVector(v []float32, a float32) {
	s := archsimd.BroadcastFloat32x4(a)
	i := 0
	for ; i+4 <= len(v); i += 4 {
		store4(load4(v, i).Mul(s), v, i)
	}
	for ; i < len(v); i++ {
		v[i] *= a
	}
}

// rotateHalves applies RoPE to one head: x1' = x1*c - x2*s, x2' = x2*c + x1*s.
func rotateHalves(x1, x2, cos, sin []float32) {
	x2, cos, sin = x2[:len(x1)], cos[:len(x1)], sin[:len(x1)]
	i := 0
	for ; i+4 <= len(x1); i += 4 {
		a, b := load4(x1, i), load4(x2, i)
		c, s := load4(cos, i), load4(sin, i)
		store4(a.Mul(c).Sub(b.Mul(s)), x1, i)
		store4(b.Mul(c).Add(a.Mul(s)), x2, i)
	}
	for ; i < len(x1); i++ {
		a, b := x1[i], x2[i]
		x1[i] = a*cos[i] - b*sin[i]
		x2[i] = b*cos[i] + a*sin[i]
	}
}

// softmaxScaled replaces row with softmax(row*scale).
func softmaxScaled(row []float32, scale float32) {
	m := float32(math.Inf(-1))
	for _, v := range row {
		m = max(m, v)
	}
	m *= scale
	s, mv := archsimd.BroadcastFloat32x4(scale), archsimd.BroadcastFloat32x4(m)
	var acc archsimd.Float32x4
	i := 0
	for ; i+4 <= len(row); i += 4 {
		e := exp4(load4(row, i).Mul(s).Sub(mv))
		store4(e, row, i)
		acc = acc.Add(e)
	}
	sum := sum4(acc)
	for ; i < len(row); i++ {
		row[i] = expNonPositive32(row[i]*scale - m)
		sum += row[i]
	}
	scaleVector(row, 1/sum)
}
