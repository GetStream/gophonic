// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import "sync"

// forcePortable routes every multiplication through the portable kernel. It
// is set only by tests, before any workspace is created.
var forcePortable bool

var (
	f16TableOnce sync.Once
	f16Table     []float32
)

// prepareF16Table is called while loading FP16 weights on portable CPUs.
// Keeping this allocation in model setup leaves the inference call allocation-free.
func prepareF16Table() {
	f16TableOnce.Do(func() {
		f16Table = make([]float32, 1<<16)
		for h := range f16Table {
			f16Table[h] = f16ToF32(uint16(h))
		}
	})
}

// portableMulPanels is the kernel for CPUs without SME. It decodes each
// weight panel to FP32 once, then accumulates every packed row against it in
// increasing K order.
func portableMulPanels(dst []float32, stride int, ws *Workspace, w *Weights, p0, p1 int, panel []float32) {
	if w.h != nil {
		prepareF16Table()
	}
	rows := ws.rows
	var acc [ActivationRows][OutputPanel]float32
	for p := p0; p < p1; p++ {
		base := p * w.pairs * 2 * OutputPanel
		for k := range w.k {
			src := base + (k/2)*2*OutputPanel + k%2
			out := panel[k*OutputPanel : (k+1)*OutputPanel]
			if w.h != nil {
				for c := range out {
					out[c] = f16Table[w.h[src+2*c]]
				}
			} else {
				for c := range out {
					out[c] = float32(w.q[src+2*c])
				}
			}
		}
		for r := range rows {
			clear(acc[r][:])
		}
		for k := range w.k {
			wrow := (*[OutputPanel]float32)(panel[k*OutputPanel:])
			a := ws.act32[k*ActivationRows : k*ActivationRows+rows]
			for r, av := range a {
				fmaPanel(&acc[r], wrow, av)
			}
		}
		col0 := p * OutputPanel
		cols := min(OutputPanel, w.n-col0)
		for r := range rows {
			out := dst[r*stride+col0 : r*stride+col0+cols]
			inv := ws.rowInverse[r]
			for c := range out {
				out[c] = acc[r][c] * w.scales[col0+c] * inv
			}
		}
	}
}

// ForcePortableForTesting routes later workspaces and multiplications through
// the portable kernel, so tests can exercise it on SME machines. It must not
// be called while any multiplication is running.
func ForcePortableForTesting(on bool) { forcePortable = on }
