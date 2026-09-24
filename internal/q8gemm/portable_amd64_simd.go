// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package q8gemm

import "simd/archsimd"

// fmaPanel retains the existing packed weight layout and reduction order.
// Eight adjacent output columns share one AVX2/FMA operation.
func fmaPanel(acc, w *[OutputPanel]float32, a float32) {
	s := archsimd.BroadcastFloat32x8(a)
	for c := 0; c < OutputPanel; c += 8 {
		wv := archsimd.LoadFloat32x8(w[c:])
		av := archsimd.LoadFloat32x8(acc[c:])
		wv.MulAdd(s, av).StoreArray((*[8]float32)(acc[c : c+8]))
	}
}
