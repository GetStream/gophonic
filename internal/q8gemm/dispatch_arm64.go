// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64

package q8gemm

func usingSME() bool { return smeEnabled }

func mulSME(dst, activation []float32, rows int, w *Weights) bool {
	if !smeEnabled {
		return false
	}
	retries := smeMul16x64(&w.q[0], w.k, w.panels, &activation[0], &dst[0], w.n, rows)
	if retries != 0 {
		smeRetries.Add(uint64(retries))
	}
	applyScales(dst, rows, w)
	return true
}
