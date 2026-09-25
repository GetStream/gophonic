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

func mulPanelsI8SME(dst []float32, stride int, ws *WorkspaceI8, w *WeightsI8, p0, p1 int) bool {
	if !usingSME() {
		return false
	}
	col := p0 * OutputPanel
	retries := smeMulI8(&w.q[p0*w.quads*4*OutputPanel], w.quads, p1-p0, &ws.activation[0],
		&dst[col], w.n-col, ws.rows, 4*stride, &w.scales[col], &ws.rowScale[0])
	if retries != 0 {
		smeRetries.Add(uint64(retries))
	}
	return true
}

func mulRowI8SME(dst []float32, ws *WorkspaceI8, w *WeightsI8, p0, p1 int) bool {
	if !usingSME() {
		return false
	}
	col := p0 * OutputPanel
	retries := smeRowI8(&w.q[p0*w.quads*4*OutputPanel], w.k/32, p1-p0, &ws.row[0],
		&dst[col], w.n-col, &w.scales[col], &ws.rowScale[0])
	if retries != 0 {
		smeRetries.Add(uint64(retries))
	}
	return true
}

func mulRowF16SME(dst []float32, ws *Workspace, w *Weights, p0, p1 int) bool {
	if !usingSME() {
		return false
	}
	col := p0 * OutputPanel
	retries := smeRowF16(&w.h[p0*w.pairs*2*OutputPanel], w.k/16, p1-p0, &ws.row[0],
		&dst[col], w.n-col, &w.scales[col], &ws.rowInverse[0])
	if retries != 0 {
		smeRetries.Add(uint64(retries))
	}
	return true
}

// mulRowsF16SME keeps one panel cache-local across all rows. Pairs share
// weight loads inside ZA; an odd final row uses the unchanged row kernel.
func mulRowsF16SME(dst []float32, stride int, ws *Workspace, w *Weights, p0, p1 int) bool {
	if !usingSME() {
		return false
	}
	var retries int
	for panel := p0; panel < p1; panel++ {
		col := panel * OutputPanel
		weight := &w.h[panel*w.pairs*2*OutputPanel]
		row := 0
		for ; row+1 < ws.rows; row += 2 {
			retries += smeRows2F16(weight, w.k/16, 1, &ws.activation[row*w.k],
				&dst[row*stride+col], w.n-col, 4*stride, &w.scales[col], &ws.rowInverse[row])
		}
		if row < ws.rows {
			retries += smeRowF16(weight, w.k/16, 1, &ws.activation[row*w.k],
				&dst[row*stride+col], w.n-col, &w.scales[col], &ws.rowInverse[row])
		}
	}
	if retries != 0 {
		smeRetries.Add(uint64(retries))
	}
	return true
}
