// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import "math"

// WeightsI8 holds signed-int8 rows for the int8×int8 path, packed as
// [output panel][K/4][64][4]: four consecutive K values of one output column
// are adjacent, matching the four-way integer outer product. Each output row
// keeps an FP32 scale. Padding bytes, including a partial final K group,
// are zero.
type WeightsI8 struct {
	k, quads, n, panels int
	q                   []int8
	scales              []float32
}

// NewWeightsI8 allocates packed storage for an N-by-K row-major int8 matrix.
func NewWeightsI8(k, n int) (*WeightsI8, error) {
	if k < 0 || n < 0 || n > math.MaxInt-(OutputPanel-1) || k > math.MaxInt-3 {
		return nil, ErrDimensions
	}
	panels := (n + OutputPanel - 1) / OutputPanel
	quads := (k + 3) / 4
	if panels != 0 && quads > math.MaxInt/panels/(4*OutputPanel) {
		return nil, ErrDimensions
	}
	return &WeightsI8{k: k, quads: quads, n: n, panels: panels,
		q: make([]int8, panels*quads*4*OutputPanel), scales: make([]float32, n)}, nil
}

// Pack copies row-major int8 values and FP32 per-row scales.
func (w *WeightsI8) Pack(q []int8, scales []float32) error {
	if w == nil || len(q) != w.k*w.n || len(scales) != w.n {
		return ErrDimensions
	}
	for _, s := range scales {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			return ErrNonFinite
		}
	}
	copy(w.scales, scales)
	clear(w.q)
	for row := range w.n {
		base := (row/OutputPanel)*w.quads*4*OutputPanel + (row%OutputPanel)*4
		for k, v := range q[row*w.k : (row+1)*w.k] {
			w.q[base+(k/4)*4*OutputPanel+k%4] = v
		}
	}
	return nil
}

// Dims returns the logical K and N dimensions.
func (w *WeightsI8) Dims() (k, n int) { return w.k, w.n }

// Panels returns the number of 64-column output panels.
func (w *WeightsI8) Panels() int { return w.panels }

// Bytes reports packed weight and scale bytes.
func (w *WeightsI8) Bytes() int { return len(w.q) + 4*len(w.scales) }

func (w *WeightsI8) at(row, k int) int8 {
	return w.q[(row/OutputPanel)*w.quads*4*OutputPanel+(k/4)*4*OutputPanel+(row%OutputPanel)*4+k%4]
}

// WorkspaceI8 owns one 16-row activation tile quantized to int8 per row:
// x ≈ q·s with s = max|x|/127 and q rounded to nearest, ties to even. The
// integer products are exact; each output is int32 · weight scale · s.
// Prepare, SetRowScale, and PackRange fill a tile; afterwards any number of
// goroutines may call MulPanelsI8 concurrently for disjoint panel ranges.
type WorkspaceI8 struct {
	activation []int8 // [K/4][16][4]
	rowScale   [ActivationRows]float32
	rowInverse [ActivationRows]float32
	k, rows    int
}

// NewWorkspaceI8 allocates activation scratch for K-wide projections.
func NewWorkspaceI8(k int) (*WorkspaceI8, error) {
	if k < 0 || k > int(^uint(0)>>3)/ActivationRows {
		return nil, ErrDimensions
	}
	return &WorkspaceI8{activation: make([]int8, ((k+3)/4)*4*ActivationRows)}, nil
}

// Prepare records the next tile's shape and clears its row scales.
func (ws *WorkspaceI8) Prepare(rows, k int) error {
	if ws == nil || rows < 0 || rows > ActivationRows || k < 0 || len(ws.activation) < ((k+3)/4)*4*ActivationRows {
		return ErrDimensions
	}
	ws.k, ws.rows = k, rows
	clear(ws.rowScale[:])
	clear(ws.rowInverse[:])
	return nil
}

// SetRowScale sets row's quantization step from its largest magnitude (see
// MaxAbs). Calls for different rows may run concurrently.
func (ws *WorkspaceI8) SetRowScale(row int, maxAbs float32) {
	if maxAbs > 0 && !math.IsInf(float64(maxAbs), 0) && maxAbs == maxAbs {
		ws.rowScale[row], ws.rowInverse[row] = maxAbs/127, 127/maxAbs
		return
	}
	ws.rowScale[row], ws.rowInverse[row] = 0, 0
}

// PackRange quantizes input columns [k0,k1) of the prepared tile, reading
// rows stride values apart. k0 and k1 must be multiples of 4 unless k1 == K.
// Disjoint ranges may be packed concurrently.
func (ws *WorkspaceI8) PackRange(x []float32, stride, k0, k1 int) error {
	if ws == nil || k0 < 0 || k1 > ws.k || k0 > k1 || k0%4 != 0 || (k1%4 != 0 && k1 != ws.k) ||
		stride < ws.k || (ws.rows > 0 && len(x) < (ws.rows-1)*stride+ws.k) {
		return ErrDimensions
	}
	q0, q1 := k0/4, (k1+3)/4
	for row := range ws.rows {
		inv := ws.rowInverse[row]
		src := x[row*stride:]
		full := q1 - q0
		if k1 == ws.k && ws.k%4 != 0 {
			full-- // the final partial group is padded below
		}
		done := packRowQuads(ws.activation[q0*4*ActivationRows+row*4:], src[k0:], full, inv)
		for quad := q0 + done; quad < q1; quad++ {
			out := ws.activation[quad*4*ActivationRows+row*4:]
			for t := range 4 {
				k := quad*4 + t
				out[t] = 0
				if k < ws.k {
					out[t] = quantize(src[k] * inv)
				}
			}
		}
	}
	if ws.rows < ActivationRows {
		for quad := q0; quad < q1; quad++ {
			clear(ws.activation[quad*4*ActivationRows+ws.rows*4 : (quad+1)*4*ActivationRows])
		}
	}
	return nil
}

// quantize rounds to nearest, ties to even, and saturates to int8, exactly
// like the NEON FCVTNS/SQXTN sequence.
func quantize(v float32) int8 {
	r := math.RoundToEven(float64(v))
	return int8(max(-128, min(127, r)))
}

// MulPanelsI8 computes output columns [p0*64, min(p1*64, N)) of
// dst[row*stride+col] = (Σ q_a·q_w) · weightScale[col] · rowScale[row]
// for the rows last packed into ws. Other columns are untouched, so disjoint
// panel ranges may run concurrently. It does not allocate.
func MulPanelsI8(dst []float32, stride int, ws *WorkspaceI8, w *WeightsI8, p0, p1 int) error {
	if w == nil || ws == nil || ws.k != w.k || p0 < 0 || p1 > w.panels || p0 > p1 || stride < w.n {
		return ErrDimensions
	}
	if ws.rows == 0 || p0 == p1 {
		return nil
	}
	if len(dst) < (ws.rows-1)*stride+w.n {
		return ErrDimensions
	}
	if w.k == 0 {
		for r := range ws.rows {
			clear(dst[r*stride+p0*OutputPanel : r*stride+min(p1*OutputPanel, w.n)])
		}
		return nil
	}
	if !mulPanelsI8SME(dst, stride, ws, w, p0, p1) {
		scalarMulPanelsI8(dst, stride, ws, w, p0, p1)
	}
	return nil
}

// scalarMulPanelsI8 is the exact integer oracle and portable fallback.
func scalarMulPanelsI8(dst []float32, stride int, ws *WorkspaceI8, w *WeightsI8, p0, p1 int) {
	for col := p0 * OutputPanel; col < min(p1*OutputPanel, w.n); col++ {
		for row := range ws.rows {
			var sum int32
			for k := range w.k {
				a := ws.activation[(k/4)*4*ActivationRows+row*4+k%4]
				sum += int32(a) * int32(w.at(col, k))
			}
			dst[row*stride+col] = float32(sum) * w.scales[col] * ws.rowScale[row]
		}
	}
}
