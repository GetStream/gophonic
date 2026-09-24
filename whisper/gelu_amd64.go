// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whisper

import "simd/archsimd"

func applyGELU(values []float32) {
	one := archsimd.BroadcastFloat32x4(1)
	half := archsimd.BroadcastFloat32x4(0.5)
	sixth := archsimd.BroadcastFloat32x4(1.0 / 6)
	shift := archsimd.BroadcastFloat32x4(65536)
	shiftBits := shift.ToBits()
	limit := archsimd.BroadcastFloat32x4(8)
	i := 0
	for ; i+4 <= len(values); i += 4 {
		x := archsimd.LoadFloat32x4Array((*[4]float32)(values[i : i+4]))
		a := x.Abs()
		if !allMask4(a.Less(limit).ToInt32x4()) {
			applyGELUScalar(values[i : i+4])
			continue
		}
		z := a.Add(shift)
		index := z.ToBits().Sub(shiftBits)
		r := z.Sub(shift)
		d := a.Sub(r)
		t0 := geluNormalTable[index.GetElem(0)]
		t1 := geluNormalTable[index.GetElem(1)]
		t2 := geluNormalTable[index.GetElem(2)]
		t3 := geluNormalTable[index.GetElem(3)]
		q := archsimd.Float32x4{}.SetElem(0, t0[0]).SetElem(1, t1[0]).SetElem(2, t2[0]).SetElem(3, t3[0])
		density := archsimd.Float32x4{}.SetElem(0, t0[1]).SetElem(1, t1[1]).SetElem(2, t2[1]).SetElem(3, t3[1])
		curvature := r.Mul(r).Sub(one).Mul(sixth)
		series := d.MulAdd(curvature, r.Mul(half).Neg()).MulAdd(d, one)
		q = density.Mul(d).Neg().MulAdd(series, q)
		q = one.Sub(q).IfElse(x.Greater(archsimd.Float32x4{}), q)
		x.Mul(q).StoreArray((*[4]float32)(values[i : i+4]))
	}
	applyGELUScalar(values[i:])
}
