// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package q8gemm multiplies small batches of FP32 activations by row-scaled
// signed-int8 weights. Its packed form is organized for a 16-by-64 SME tile.
package q8gemm

import (
	"errors"
	"math"
)

const (
	// OutputPanel is the number of output columns in one packed weight panel.
	OutputPanel = 64
	// ActivationRows is the number of activation rows consumed by one tile.
	ActivationRows = 16
	// OutputPanelBytes is the packed weight bytes for one K position.
	OutputPanelBytes = OutputPanel
)

var (
	ErrDimensions = errors.New("q8gemm: invalid dimensions or buffer length")
	ErrNonFinite  = errors.New("q8gemm: non-finite row scale")
)

// Weights owns signed-int8 rows packed as [output panel][K][64], plus the
// original FP32 scale for each logical output row. Padding bytes are zero.
// Pack is a one-time model setup operation; MulInto only reads this storage.
type Weights struct {
	k, n, panels int
	q            []int8
	scales       []float32
}

// NewWeights allocates packed storage for an N-by-K row-major Q8 matrix.
func NewWeights(k, n int) (*Weights, error) {
	if k < 0 || n < 0 || n > math.MaxInt-(OutputPanel-1) {
		return nil, ErrDimensions
	}
	panels := (n + OutputPanel - 1) / OutputPanel
	if panels != 0 && k > math.MaxInt/panels/OutputPanel {
		return nil, ErrDimensions
	}
	return &Weights{
		k:      k,
		n:      n,
		panels: panels,
		q:      make([]int8, panels*k*OutputPanel),
		scales: make([]float32, n),
	}, nil
}

// Pack copies row-major Q8 values and FP32 per-row scales into the SME panel
// layout. It reuses the receiver's storage on repeated calls.
func (w *Weights) Pack(q []int8, scales []float32) error {
	if w == nil || len(q) != w.k*w.n || len(scales) != w.n {
		return ErrDimensions
	}
	for _, scale := range scales {
		if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
			return ErrNonFinite
		}
	}
	copy(w.scales, scales)
	clear(w.q)
	for panel := 0; panel < w.panels; panel++ {
		panelBase := panel * w.k * OutputPanel
		for k := 0; k < w.k; k++ {
			dst := w.q[panelBase+k*OutputPanel : panelBase+(k+1)*OutputPanel]
			for col := range OutputPanel {
				row := panel*OutputPanel + col
				if row < w.n {
					dst[col] = q[row*w.k+k]
				}
			}
		}
	}
	return nil
}

// Dims returns the logical K and N dimensions.
func (w *Weights) Dims() (k, n int) {
	if w == nil {
		return 0, 0
	}
	return w.k, w.n
}

// Bytes reports packed Q8 weight bytes and FP32 row-scale bytes.
func (w *Weights) Bytes() int {
	if w == nil {
		return 0
	}
	return len(w.q) + 4*len(w.scales)
}
