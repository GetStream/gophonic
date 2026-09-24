// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64

package q8gemm

func usingSME() bool { return smeEnabled && !forcePortable }

func mulPanelsSME(dst []float32, stride int, ws *Workspace, w *Weights, p0, p1 int) bool {
	if !usingSME() {
		return false
	}
	col := p0 * OutputPanel
	var retries int
	if w.h != nil {
		retries = smeMulF16W(&w.h[p0*w.pairs*2*OutputPanel], w.pairs, p1-p0, &ws.activation[0],
			&dst[col], w.n-col, ws.rows, 4*stride, &w.scales[col], &ws.rowInverse[0])
	} else {
		retries = smeMulF16(&w.q[p0*w.pairs*2*OutputPanel], w.pairs, p1-p0, &ws.activation[0],
			&dst[col], w.n-col, ws.rows, 4*stride, &w.scales[col], &ws.rowInverse[0])
	}
	if retries != 0 {
		smeRetries.Add(uint64(retries))
	}
	return true
}

func retryCount() uint64 { return smeRetries.Load() }
