// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

func scalarMul(dst, activation []float32, rows int, w *Weights) {
	for panel := range w.panels {
		panelBase := panel * w.k * OutputPanel
		for colBase := 0; colBase < OutputPanel; colBase += 16 {
			col := panel*OutputPanel + colBase
			if col >= w.n {
				break
			}
			var sums [ActivationRows][16]float32
			for kk := range w.k {
				weights := w.q[panelBase+kk*OutputPanel+colBase : panelBase+kk*OutputPanel+colBase+16]
				acts := activation[kk*ActivationRows : (kk+1)*ActivationRows]
				for lane, q := range weights {
					weight := float32(q)
					for row := range ActivationRows {
						sums[row][lane] += acts[row] * weight
					}
				}
			}
			for row := range rows {
				out := dst[row*w.n+col : row*w.n+col+min(16, w.n-col)]
				for lane := range out {
					out[lane] = sums[row][lane] * w.scales[col+lane]
				}
			}
		}
	}
}
