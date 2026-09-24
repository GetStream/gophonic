// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

// Workspace owns one transposed 16-row activation tile. It can be reused by
// serial calls to one or more MulInto invocations and must not be shared by
// concurrent calls.
type Workspace struct {
	activation []float32 // [K][16]
}

// Available reports whether this process can use the 512-bit SME kernel. It is
// safe to call on any architecture; false means callers should choose another
// optimized kernel when one is available.
func Available() bool { return usingSME() }

// NewWorkspace allocates enough activation scratch for a K-wide projection.
func NewWorkspace(k int) (*Workspace, error) {
	if k < 0 || k > int(^uint(0)>>1)/ActivationRows {
		return nil, ErrDimensions
	}
	return &Workspace{activation: make([]float32, k*ActivationRows)}, nil
}

// MulInto computes dst[M,N] = x[M,K] * (Q8[N,K] * rowScale[N]). The source
// activation and destination are row-major. At most 16 input rows are accepted.
// The weights are packed once by Weights.Pack. The caller owns the workspace;
// warmed calls allocate nothing. FP32 activations and original FP32 weight
// scales are preserved. The reduction order is implementation-dependent and
// may differ slightly from a scalar or NEON sum.
func MulInto(dst, x []float32, rows int, w *Weights, ws *Workspace) error {
	if w == nil || ws == nil || rows < 0 || rows > ActivationRows || len(dst) < rows*w.n || len(x) < rows*w.k || len(ws.activation) < w.k*ActivationRows {
		return ErrDimensions
	}
	if rows == 0 || w.n == 0 {
		return nil
	}
	if w.k == 0 {
		for r := range rows {
			clear(dst[r*w.n : (r+1)*w.n])
		}
		return nil
	}
	activation := ws.activation[:w.k*ActivationRows]
	packActivations(activation, x, rows, w.k)
	if mulSME(dst, activation, rows, w) {
		return nil
	}
	scalarMul(dst, activation, rows, w)
	return nil
}

// packActivations transposes at most 16 rows and zero-pads unused lanes.
func packActivations(dst, src []float32, rows, k int) {
	for kk := range k {
		out := dst[kk*ActivationRows : (kk+1)*ActivationRows]
		for row := range rows {
			out[row] = src[row*k+kk]
		}
		clear(out[rows:])
	}
}

func applyScales(dst []float32, rows int, w *Weights) {
	for row := range rows {
		out := dst[row*w.n : (row+1)*w.n]
		for col, value := range out {
			out[col] = value * w.scales[col]
		}
	}
}
