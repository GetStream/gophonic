// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whisper

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

const layerNormAccelerated = true

// The public callers validate positive lengths and every backing slice.
// Float64 statistics preserve the encoder's cancellation-sensitive contract.
func layerNormNEON(src, dst, gamma, beta *float32, n int) {
	x := unsafe.Slice(src, n)
	y := unsafe.Slice(dst, n)
	g := unsafe.Slice(gamma, n)
	b := unsafe.Slice(beta, n)
	var sum0, sum1 archsimd.Float64x4
	for i := 0; i < n; i += 8 {
		v := archsimd.LoadFloat32x8(x[i:])
		sum0 = sum0.Add(v.GetLo().ConvertToFloat64())
		sum1 = sum1.Add(v.GetHi().ConvertToFloat64())
	}
	mean := reduceFloat64x4(sum0.Add(sum1)) / float64(n)
	m := archsimd.BroadcastFloat64x4(mean)
	var var0, var1 archsimd.Float64x4
	for i := 0; i < n; i += 8 {
		v := archsimd.LoadFloat32x8(x[i:])
		d0 := v.GetLo().ConvertToFloat64().Sub(m)
		d1 := v.GetHi().ConvertToFloat64().Sub(m)
		var0 = d0.MulAdd(d0, var0)
		var1 = d1.MulAdd(d1, var1)
	}
	variance := reduceFloat64x4(var0.Add(var1)) / float64(n)
	inv := archsimd.BroadcastFloat64x4(1 / math.Sqrt(variance+1e-5))
	for i := 0; i < n; i += 8 {
		v := archsimd.LoadFloat32x8(x[i:])
		gv := archsimd.LoadFloat32x8(g[i:])
		bv := archsimd.LoadFloat32x8(b[i:])
		y0 := v.GetLo().ConvertToFloat64().Sub(m).Mul(inv).MulAdd(gv.GetLo().ConvertToFloat64(), bv.GetLo().ConvertToFloat64())
		y1 := v.GetHi().ConvertToFloat64().Sub(m).Mul(inv).MulAdd(gv.GetHi().ConvertToFloat64(), bv.GetHi().ConvertToFloat64())
		y0.ConvertToFloat32().StoreArray((*[4]float32)(y[i : i+4]))
		y1.ConvertToFloat32().StoreArray((*[4]float32)(y[i+4 : i+8]))
	}
}

func reduceFloat64x4(v archsimd.Float64x4) float64 {
	var lanes [4]float64
	v.StoreArray(&lanes)
	return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

// The row, add, and output ranges are independent except that dst may alias
// add. Finish reading add into row before normalization writes dst.
func residualNormNEON(row, dst, gamma, beta *float32, n int, add, bias *float32) {
	r := unsafe.Slice(row, n)
	a := unsafe.Slice(add, n)
	if bias != nil {
		b := unsafe.Slice(bias, n)
		for i := 0; i < n; i += 8 {
			v := archsimd.LoadFloat32x8(r[i:])
			s := archsimd.LoadFloat32x8(a[i:]).Add(archsimd.LoadFloat32x8(b[i:]))
			v.Add(s).StoreArray((*[8]float32)(r[i : i+8]))
		}
	} else {
		for i := 0; i < n; i += 8 {
			v := archsimd.LoadFloat32x8(r[i:])
			v.Add(archsimd.LoadFloat32x8(a[i:])).StoreArray((*[8]float32)(r[i : i+8]))
		}
	}
	layerNormNEON(row, dst, gamma, beta, n)
}

func attnPrepNEON(q, k, v, qbias, vbias *float32, n int, scale float32) {
	qs, ks, vs := unsafe.Slice(q, n), unsafe.Slice(k, n), unsafe.Slice(v, n)
	qb, vb := unsafe.Slice(qbias, n), unsafe.Slice(vbias, n)
	s := archsimd.BroadcastFloat32x8(scale)
	i := 0
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8(qs[i:]).Add(archsimd.LoadFloat32x8(qb[i:])).Mul(s).StoreArray((*[8]float32)(qs[i : i+8]))
		archsimd.LoadFloat32x8(ks[i:]).Mul(s).StoreArray((*[8]float32)(ks[i : i+8]))
		archsimd.LoadFloat32x8(vs[i:]).Add(archsimd.LoadFloat32x8(vb[i:])).StoreArray((*[8]float32)(vs[i : i+8]))
	}
	for ; i < n; i++ {
		qs[i] = (qs[i] + qb[i]) * scale
		ks[i] *= scale
		vs[i] += vb[i]
	}
}

// A maximum with the running value as the second operand ignores NaNs in the
// loaded vector under x86 MAXPS semantics. The scalar tail has the same rule.
func maxNumNEON(x *float32, n int) float32 {
	values := unsafe.Slice(x, n)
	best := archsimd.BroadcastFloat32x8(float32(math.Inf(-1)))
	i := 0
	for ; i+8 <= n; i += 8 {
		best = archsimd.LoadFloat32x8(values[i:]).Max(best)
	}
	var lanes [8]float32
	best.StoreArray(&lanes)
	maximum := float32(math.Inf(-1))
	for _, v := range lanes {
		if v > maximum {
			maximum = v
		}
	}
	for ; i < n; i++ {
		if values[i] > maximum {
			maximum = values[i]
		}
	}
	return maximum
}
