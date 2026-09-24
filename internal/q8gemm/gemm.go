// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import "math"

// Workspace owns one 16-row activation tile in FP16, laid out [K/2][16][2],
// plus one exact power-of-two scale per row. Each row is scaled so its largest
// magnitude lies in [2^14, 2^15) before rounding to FP16, which keeps every
// value finite and gives each element FP16's 11-bit relative precision
// (finer than the BF16 activations of the reference Qwen3 runtime). Products
// and sums are accumulated in FP32; the row scale is removed exactly at the
// output.
//
// Prepare, SetRowScale, and PackRange fill a tile; afterwards any number of
// goroutines may call MulPanels concurrently for disjoint panel ranges.
type Workspace struct {
	activation []uint16
	act32      []float32 // [K][16] FP32 copy of activation, only without SME
	scratch    *Scratch  // MulInto's own scratch, only without SME
	rowScale   [ActivationRows]float32 // multiplies inputs before FP16 rounding
	rowInverse [ActivationRows]float32 // multiplies outputs
	k, rows    int
}

// Scratch holds one goroutine's weight-decoding buffer for the portable path
// used on CPUs without SME. NewScratch returns nil when SME is available.
type Scratch struct {
	panel []float32 // [K][64] decoded weights of one panel
}

// NewScratch allocates portable-path scratch for K-wide projections, or
// returns nil when the SME kernel will be used.
func NewScratch(k int) *Scratch {
	if usingSME() || k < 0 {
		return nil
	}
	return &Scratch{panel: make([]float32, k*OutputPanel)}
}

// Available reports whether this process can use the 512-bit SME kernel. It is
// safe to call on any architecture; false means callers should choose another
// optimized kernel when one is available.
func Available() bool { return usingSME() }

// NewWorkspace allocates enough activation scratch for a K-wide projection.
func NewWorkspace(k int) (*Workspace, error) {
	if k < 0 || k > int(^uint(0)>>2)/ActivationRows {
		return nil, ErrDimensions
	}
	ws := &Workspace{activation: make([]uint16, ((k+1)/2)*2*ActivationRows)}
	if !usingSME() {
		ws.act32 = make([]float32, k*ActivationRows)
		ws.scratch = NewScratch(k)
	}
	return ws, nil
}

// Pack prepares and packs rows (at most 16) of the row-major x[rows,k].
func (ws *Workspace) Pack(x []float32, rows, k int) error {
	if err := ws.Prepare(rows, k); err != nil {
		return err
	}
	if len(x) < rows*k {
		return ErrDimensions
	}
	for row := range rows {
		ws.SetRowScale(row, MaxAbs(x[row*k:(row+1)*k]))
	}
	return ws.PackRange(x, k, 0, k)
}

// Prepare records the shape of the next tile and resets its row scales. It
// must not run concurrently with any other method on the workspace.
func (ws *Workspace) Prepare(rows, k int) error {
	if ws == nil || rows < 0 || rows > ActivationRows || k < 0 || len(ws.activation) < ((k+1)/2)*2*ActivationRows {
		return ErrDimensions
	}
	ws.k, ws.rows = k, rows
	for i := range ws.rowScale {
		ws.rowScale[i], ws.rowInverse[i] = 1, 1
	}
	return nil
}

// SetRowScale sets row's exact power-of-two scale from the largest magnitude
// in that row (see MaxAbs). Call it for every prepared row before PackRange;
// calls for different rows may run concurrently.
func (ws *Workspace) SetRowScale(row int, maxAbs float32) {
	e := 0
	if maxAbs > 0 && !math.IsInf(float64(maxAbs), 0) && !math.IsNaN(float64(maxAbs)) {
		_, exp := math.Frexp(float64(maxAbs)) // maxAbs in [2^(exp-1), 2^exp)
		e = min(max(15-exp, -100), 100)
	}
	ws.rowScale[row] = float32(math.Ldexp(1, e))
	ws.rowInverse[row] = float32(math.Ldexp(1, -e))
}

// MaxAbs returns the largest magnitude in x, or NaN/Inf if x holds one.
func MaxAbs(x []float32) float32 {
	m, done := maxAbsPrefix(x)
	if m != m {
		return m
	}
	for _, v := range x[done:] {
		if v != v {
			return v
		}
		m = max(m, math.Float32frombits(math.Float32bits(v)&^(1<<31)))
	}
	return m
}

// PackRange converts input columns [k0,k1) of the prepared tile to scaled
// FP16, reading rows stride values apart. k0 and k1 must be multiples of
// PackAlign unless k1 == K. Disjoint ranges may be packed concurrently.
func (ws *Workspace) PackRange(x []float32, stride, k0, k1 int) error {
	if ws == nil || k0 < 0 || k1 > ws.k || k0 > k1 || k0%PackAlign != 0 || (k1%PackAlign != 0 && k1 != ws.k) ||
		stride < ws.k || (ws.rows > 0 && len(x) < (ws.rows-1)*stride+ws.k) {
		return ErrDimensions
	}
	rows := ws.rows
	p0, p1 := k0/2, (k1+1)/2
	// Rows are converted independently; each row's pairs sit 64 bytes apart.
	for row := range rows {
		scale := ws.rowScale[row]
		src := x[row*stride:]
		dst := ws.activation[p0*2*ActivationRows+row*2:]
		done := 0
		if k1 == ws.k && ws.k%2 == 1 {
			// The final odd element needs zero padding; keep it scalar.
			done = packRowPairs(dst, src[k0:], p1-p0-1, scale)
		} else {
			done = packRowPairs(dst, src[k0:], p1-p0, scale)
		}
		for pair := p0 + done; pair < p1; pair++ {
			kk := pair * 2
			out := ws.activation[pair*2*ActivationRows+row*2:]
			out[0] = f32ToF16(src[kk] * scale)
			if kk+1 < ws.k {
				out[1] = f32ToF16(src[kk+1] * scale)
			} else {
				out[1] = 0
			}
		}
	}
	if rows < ActivationRows {
		for pair := p0; pair < p1; pair++ {
			clear(ws.activation[pair*2*ActivationRows+rows*2 : (pair+1)*2*ActivationRows])
		}
	}
	if ws.act32 != nil {
		// The portable kernel reads the same FP16-rounded values as FP32.
		for kk := k0; kk < k1; kk++ {
			out := ws.act32[kk*ActivationRows : (kk+1)*ActivationRows]
			for row := range rows {
				out[row] = f16ToF32(ws.activation[(kk/2)*2*ActivationRows+row*2+kk%2])
			}
			clear(out[rows:])
		}
	}
	return nil
}

// MulPanels computes output columns [p0*64, min(p1*64, N)) of
// dst[row*stride+col] = packed[row,:] · W[col,:] * rowScale[col]
// for the rows last packed into ws. Other columns of dst are untouched, so
// disjoint panel ranges may run concurrently, each with its own scratch from
// NewScratch (nil when SME is available). It does not allocate.
func MulPanels(dst []float32, stride int, ws *Workspace, w *Weights, p0, p1 int, scratch *Scratch) error {
	if w == nil || ws == nil || ws.k != w.k || p0 < 0 || p1 > w.panels || p0 > p1 || stride < w.n {
		return ErrDimensions
	}
	rows := ws.rows
	if rows == 0 || p0 == p1 {
		return nil
	}
	if len(dst) < (rows-1)*stride+w.n {
		return ErrDimensions
	}
	if w.k == 0 {
		for r := range rows {
			clear(dst[r*stride+p0*OutputPanel : r*stride+min(p1*OutputPanel, w.n)])
		}
		return nil
	}
	if mulPanelsSME(dst, stride, ws, w, p0, p1) {
		return nil
	}
	if scratch == nil || len(scratch.panel) < w.k*OutputPanel || len(ws.act32) < w.k*ActivationRows {
		return ErrDimensions
	}
	portableMulPanels(dst, stride, ws, w, p0, p1, scratch.panel[:w.k*OutputPanel])
	return nil
}

// MulInto computes dst[M,N] = x[M,K] * (Q8[N,K] * rowScale[N]) for at most
// 16 row-major input rows, with the activation precision described on
// Workspace. The caller owns the workspace; warmed calls allocate nothing.
func MulInto(dst, x []float32, rows int, w *Weights, ws *Workspace) error {
	if w == nil || ws == nil || rows < 0 || rows > ActivationRows || len(dst) < rows*w.n || len(x) < rows*w.k {
		return ErrDimensions
	}
	if rows == 0 || w.n == 0 {
		return nil
	}
	if err := ws.Pack(x[:rows*w.k], rows, w.k); err != nil {
		return err
	}
	return MulPanels(dst, w.n, ws, w, 0, w.panels, ws.scratch)
}

// Retries reports how many SME output panels were recomputed because a signal
// cleared streaming vector registers. It is a process-wide diagnostic counter.
func Retries() uint64 { return retryCount() }

// f32ToF16 rounds to the nearest FP16 value, ties to even. Callers scale
// inputs so they never exceed FP16's finite range; NaN stays NaN.
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case b&0x7fffffff > 0x7f800000:
		return sign | 0x7e00
	case exp >= 31:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		half := mant >> shift
		rem := mant & (1<<shift - 1)
		halfway := uint32(1) << (shift - 1)
		if rem > halfway || rem == halfway && half&1 == 1 {
			half++
		}
		return sign | uint16(half)
	}
	half := uint32(exp)<<10 | mant>>13
	rem := mant & 0x1fff
	if rem > 0x1000 || rem == 0x1000 && half&1 == 1 {
		half++
	}
	return sign | uint16(half)
}

// f16ToF32 widens an FP16 value exactly.
func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		return math.Float32frombits(sign) + float32(mant)*(1.0/16777216.0)*sign32(sign)
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}

func sign32(sign uint32) float32 {
	if sign != 0 {
		return -1
	}
	return 1
}
