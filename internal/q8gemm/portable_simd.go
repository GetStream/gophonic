// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package q8gemm

import "simd/archsimd"

// fmaPanel computes acc += a*w over one 64-column panel.
func fmaPanel(acc, w *[OutputPanel]float32, a float32) {
	s := archsimd.BroadcastFloat32x4(a)
	for c := 0; c < OutputPanel; c += 4 {
		wv := archsimd.LoadFloat32x4Array((*[4]float32)(w[c : c+4]))
		av := archsimd.LoadFloat32x4Array((*[4]float32)(acc[c : c+4]))
		wv.MulAdd(s, av).StoreArray((*[4]float32)(acc[c : c+4]))
	}
}
