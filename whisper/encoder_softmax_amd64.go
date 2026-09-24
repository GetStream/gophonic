// Copyright 2026 The gophonic authors
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//
// The exponential polynomial is adapted from ggml_v_expf in
// ggml/src/ggml-cpu/vec.h at whisper.cpp commit
// a664346ea5c6dddff3e61a2b7b32dd4514613f50. The pinned source credits Arm's
// optimized routines and states a maximum error of 1.45358 + 0.5 ulps.
// See testdata/softmax_exp_neon.c for the independent source-shaped oracle
// and the MIT license notice.

//go:build goexperiment.simd && amd64 && amd64.v3

package whisper

import (
	"math"
	"simd/archsimd"
)

// softmaxExpInPlace evaluates exp(x-maxValue), whose finite arguments are
// nonpositive. Normal and subnormal results follow the pinned ggml NEON
// approximation; scalar tails use math.Exp. The sum retains the original
// increasing-index FP32 reduction order.
func softmaxExpInPlace(x []float32, maxValue float32) float32 {
	maximum := archsimd.BroadcastFloat32x4(maxValue)
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
	i := 0
	for ; i+4 <= len(x); i += 4 {
		input := archsimd.LoadFloat32x4Array((*[4]float32)(x[i : i+4])).Sub(maximum)
		z := input.MulAdd(log2e, magic)
		n := z.Sub(magic)
		b := n.MulAdd(negLn2High, input)
		b = n.MulAdd(negLn2Low, b)
		e := z.ToBits().ShiftAllLeft(23)
		k := e.Add(oneBits).BitsToFloat32()
		u := b.Mul(b)
		lower := p2.MulAdd(b, p1)
		upper := p4.MulAdd(b, p3)
		j := upper.MulAdd(u, lower).MulAdd(u, p0.Mul(b))
		value := j.MulAdd(k, k)
		special := n.Abs().Greater(normalLimit)
		if anyMask4(special.ToInt32x4()) {
			d := archsimd.BroadcastUint32x4(0x82000000).Masked(n.LessEqual(archsimd.Float32x4{}))
			s1 := d.Add(archsimd.BroadcastUint32x4(0x7f000000)).BitsToFloat32()
			s2 := e.Sub(d).BitsToFloat32()
			value = j.MulAdd(s2, s2).Mul(s1).IfElse(special, value)
			value = s1.Mul(s1).IfElse(n.Abs().Greater(archsimd.BroadcastFloat32x4(192)), value)
		}
		value.StoreArray((*[4]float32)(x[i : i+4]))
	}
	for ; i < len(x); i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxValue)))
	}
	var total float32
	for _, value := range x {
		total += value
	}
	return total
}
