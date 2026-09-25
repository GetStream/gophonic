// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package q8gemm multiplies small batches of activations by packed,
// row-scaled weights: signed int8 with an FP32 scale per output row, or FP16
// holding BF16 checkpoint weights exactly. Its packed form is organized for a
// 16-by-64 SME tile that multiplies FP16 activations with FP32 accumulation.
package q8gemm

import (
	"errors"
	"math"
	"unsafe"
)

const (
	// OutputPanel is the number of output columns in one packed weight panel.
	OutputPanel = 64
	// ActivationRows is the number of activation rows consumed by one tile.
	ActivationRows = 16
	// PackAlign is the K granularity of PackRange boundaries.
	PackAlign = 2
)

var (
	ErrDimensions = errors.New("q8gemm: invalid dimensions or buffer length")
	ErrNonFinite  = errors.New("q8gemm: non-finite row scale")
)

// Weights owns signed-int8 rows packed as [output panel][K/2][64][2]: each
// K pair of one output column is adjacent, matching the two-way widening
// FP16 outer product. It also keeps the original FP32 scale of each logical
// output row. Padding bytes, including an odd K's final pair, are zero.
// Pack is a one-time model setup operation; MulPanels only reads this storage.
type Weights struct {
	k, pairs, n, panels int
	q                   []int8   // int8 weights, or nil
	h                   []uint16 // FP16 weights, or nil
	scales              []float32
}

// NewWeights allocates packed storage for an N-by-K row-major Q8 matrix.
func NewWeights(k, n int) (*Weights, error) {
	if k < 0 || n < 0 || n > math.MaxInt-(OutputPanel-1) || k > math.MaxInt-1 {
		return nil, ErrDimensions
	}
	panels := (n + OutputPanel - 1) / OutputPanel
	pairs := (k + 1) / 2
	if panels != 0 && pairs > math.MaxInt/panels/(2*OutputPanel) {
		return nil, ErrDimensions
	}
	return &Weights{
		k:      k,
		pairs:  pairs,
		n:      n,
		panels: panels,
		q:      make([]int8, panels*pairs*2*OutputPanel),
		scales: make([]float32, n),
	}, nil
}

// NewWeightsF16 allocates packed FP16 storage for an N-by-K matrix, filled
// by PackBF16. It uses the same panel layout as NewWeights with two bytes
// per weight.
func NewWeightsF16(k, n int) (*Weights, error) {
	w, err := NewWeights(0, n)
	if err != nil || k < 0 || k > math.MaxInt-1 {
		return nil, ErrDimensions
	}
	w.k, w.pairs, w.q = k, (k+1)/2, nil
	if w.panels != 0 && w.pairs > math.MaxInt/w.panels/(2*OutputPanel) {
		return nil, ErrDimensions
	}
	w.h = make([]uint16, w.panels*w.pairs*2*OutputPanel)
	return w, nil
}

// WeightsF16Bytes reports bytes for NewWeightsF16Buffer, including scales.
func WeightsF16Bytes(k, n int) (int, error) {
	if k < 0 || n < 0 || k > math.MaxInt-1 || n > math.MaxInt-(OutputPanel-1) {
		return 0, ErrDimensions
	}
	panels, pairs := (n+OutputPanel-1)/OutputPanel, (k+1)/2
	if n > (math.MaxInt-63)/4 {
		return 0, ErrDimensions
	}
	scales := n * 4
	if panels != 0 && pairs > (math.MaxInt-scales-63)/panels/(4*OutputPanel) {
		return 0, ErrDimensions
	}
	return panels*pairs*4*OutputPanel + scales, nil
}

// NewWeightsF16Buffer binds packed FP16 weights and scales to caller-owned,
// four-byte-aligned zeroed storage. The caller must retain its owner through
// all Pack and Mul calls and release it only after all consumers have stopped.
func NewWeightsF16Buffer(k, n int, data []byte) (*Weights, error) {
	size, err := WeightsF16Bytes(k, n)
	if err != nil || len(data) < size {
		return nil, ErrDimensions
	}
	w := &Weights{k: k, n: n, pairs: (k + 1) / 2, panels: (n + OutputPanel - 1) / OutputPanel}
	if size == 0 {
		w.h = []uint16{}
		return w, nil
	}
	if uintptr(unsafe.Pointer(&data[0]))%4 != 0 {
		return nil, ErrDimensions
	}
	count := w.panels * w.pairs * 2 * OutputPanel
	w.h = unsafe.Slice((*uint16)(unsafe.Pointer(&data[0])), count)
	if n != 0 {
		w.scales = unsafe.Slice((*float32)(unsafe.Pointer(&data[count*2])), n)
	}
	return w, nil
}

// PackBF16 packs row-major BF16 weights (raw bits) into FP16 storage without
// rounding them. Each output row is multiplied by an exact power of two that
// puts its largest magnitude in [2^14, 2^15); the inverse becomes the row
// scale. Every BF16 value within 2^-24 of its row's largest magnitude is then
// an FP16 normal or exact subnormal, so it converts exactly; smaller values
// round. PackBF16 returns how many values were rounded. Non-finite weights are
// rejected.
func (w *Weights) PackBF16(bf16 []uint16) (rounded int, err error) {
	if w == nil || w.h == nil || len(bf16) != w.k*w.n {
		return 0, ErrDimensions
	}
	clear(w.h)
	if w.k == 0 {
		for row := range w.scales {
			w.scales[row] = 1
		}
		return 0, nil
	}
	return w.PackBF16Rows(bf16, 0)
}

// PackBF16Rows is PackBF16 for output rows [row0, row0+len(bf16)/K) of
// freshly allocated weights, so a large matrix can be packed from row
// chunks, concurrently for disjoint chunks.
func (w *Weights) PackBF16Rows(bf16 []uint16, row0 int) (rounded int, err error) {
	if w == nil || w.h == nil || w.k == 0 || len(bf16)%w.k != 0 || row0 < 0 || row0+len(bf16)/w.k > w.n {
		return 0, ErrDimensions
	}
	for i := range len(bf16) / w.k {
		row := row0 + i
		src := bf16[i*w.k : (i+1)*w.k]
		var maxAbs float32
		for _, b := range src {
			v := BF16ToF32(b)
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return 0, ErrNonFinite
			}
			maxAbs = max(maxAbs, float32(math.Abs(float64(v))))
		}
		e := 0
		if maxAbs > 0 {
			_, exp := math.Frexp(float64(maxAbs))
			e = min(max(15-exp, -100), 100)
		}
		scale := float32(math.Ldexp(1, e))
		w.scales[row] = float32(math.Ldexp(1, -e))
		panel, col := row/OutputPanel, row%OutputPanel
		base := panel * w.pairs * 2 * OutputPanel
		for k, b := range src {
			v := BF16ToF32(b) * scale
			h := f32ToF16(v)
			if f16ToF32(h) != v {
				rounded++
			}
			w.h[base+(k/2)*2*OutputPanel+col*2+k%2] = h
		}
	}
	return rounded, nil
}

// BF16ToF32 widens raw BF16 bits exactly.
func BF16ToF32(b uint16) float32 { return math.Float32frombits(uint32(b) << 16) }

// Pack copies row-major Q8 values and FP32 per-row scales into the SME panel
// layout. It reuses the receiver's storage on repeated calls.
func (w *Weights) Pack(q []int8, scales []float32) error {
	if w == nil || w.q == nil && w.k*w.n != 0 || len(q) != w.k*w.n || len(scales) != w.n {
		return ErrDimensions
	}
	for _, scale := range scales {
		if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
			return ErrNonFinite
		}
	}
	copy(w.scales, scales)
	clear(w.q)
	for row := range w.n {
		panel, col := row/OutputPanel, row%OutputPanel
		base := panel * w.pairs * 2 * OutputPanel
		src := q[row*w.k : (row+1)*w.k]
		for k, value := range src {
			w.q[base+(k/2)*2*OutputPanel+col*2+k%2] = value
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

// Bytes reports packed weight bytes and FP32 row-scale bytes.
func (w *Weights) Bytes() int {
	if w == nil {
		return 0
	}
	return len(w.q) + 2*len(w.h) + 4*len(w.scales)
}

// F16 reports whether the weights are stored as FP16.
func (w *Weights) F16() bool { return w != nil && w.h != nil }

// Panels returns the number of 64-column output panels.
func (w *Weights) Panels() int {
	if w == nil {
		return 0
	}
	return w.panels
}

// at returns the unscaled weight for output row and input k.
func (w *Weights) at(row, k int) float32 {
	i := (row/OutputPanel)*w.pairs*2*OutputPanel + (k/2)*2*OutputPanel + (row%OutputPanel)*2 + k%2
	if w.h != nil {
		return f16ToF32(w.h[i])
	}
	return float32(w.q[i])
}
