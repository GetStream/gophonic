// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

// scalarMulPanels is the portable oracle for panels [p0, p1): it widens the
// packed FP16 activations and weights exactly, accumulates in FP32, and
// applies the column and row scales like the SME kernel.
func scalarMulPanels(dst []float32, stride int, ws *Workspace, w *Weights, p0, p1 int) {
	for col := p0 * OutputPanel; col < min(p1*OutputPanel, w.n); col++ {
		for row := range ws.rows {
			var sum float32
			for k := range w.k {
				a := f16ToF32(ws.activation[(k/2)*2*ActivationRows+row*2+k%2])
				sum += a * w.at(col, k)
			}
			dst[row*stride+col] = sum * w.scales[col] * ws.rowInverse[row]
		}
	}
}
