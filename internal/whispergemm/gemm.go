// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package whispergemm provides reusable FP32 right-matrix packing and GEMM for
// Whisper projections and attention. It has no cgo or assembly dependency.
package whispergemm

import "errors"

const panelColumns = 16

var (
	ErrShape     = errors.New("whispergemm: invalid matrix dimensions, stride, or buffer length")
	ErrNilMatrix = errors.New("whispergemm: nil packed matrix")
)

// PackedB stores a K-by-N right matrix in panels of sixteen output columns.
// It owns one FP32 copy, with at most fifteen padding columns. Concurrent Mul
// calls are safe when their output buffers do not overlap. Pack must not run
// concurrently with either Pack or Mul on the same PackedB.
type PackedB struct {
	k, n int
	data []float32 // [ceil(N/16)][K][16]
}

// NewPackedB allocates a zero-filled K-by-N matrix. Pack can replace its values
// repeatedly without further allocation. Zero dimensions are supported.
func NewPackedB(k, n int) (*PackedB, error) {
	const maxInt = int(^uint(0) >> 1)
	if k < 0 || n < 0 || n > maxInt-(panelColumns-1) {
		return nil, ErrShape
	}
	columns := (n + panelColumns - 1) / panelColumns * panelColumns
	if columns != 0 && k > (maxInt/4)/columns {
		return nil, ErrShape
	}
	return &PackedB{k: k, n: n, data: make([]float32, k*columns)}, nil
}

// Dims returns the logical K and N dimensions, excluding padding.
func (b *PackedB) Dims() (k, n int) {
	if b == nil {
		return 0, 0
	}
	return b.k, b.n
}

// Pack copies a row-major K-by-N matrix, or an N-by-K matrix when transposed
// is true. stride is measured in float32 elements between source rows. The
// source may have padding and need not be vector aligned. Pack allocates no
// memory, including when repacking dynamic attention keys or values.
func (b *PackedB) Pack(src []float32, stride int, transposed bool) error {
	if b == nil {
		return ErrNilMatrix
	}
	rows, cols := b.k, b.n
	if transposed {
		rows, cols = cols, rows
	}
	if !validMatrix(src, rows, cols, stride) {
		return ErrShape
	}
	if b.k == 0 || b.n == 0 {
		return nil
	}
	for n := 0; n < b.n; n += panelColumns {
		width := min(panelColumns, b.n-n)
		panel := b.data[n*b.k : (n+panelColumns)*b.k]
		for k := 0; k < b.k; k++ {
			row := panel[k*panelColumns : (k+1)*panelColumns]
			if transposed {
				for j := 0; j < width; j++ {
					row[j] = src[(n+j)*stride+k]
				}
			} else {
				copy(row[:width], src[k*stride+n:k*stride+n+width])
			}
			clear(row[width:])
		}
	}
	return nil
}

// Mul writes C[M,N] = A[M,K] * B[K,N]. Strides are measured in elements.
// It overwrites only the logical output columns, leaving row padding intact.
// The input and output must not overlap. No alignment beyond float32 is
// required. M=0 or N=0 does nothing; K=0 writes positive zero.
//
// Mul is single-threaded and allocates no memory. Callers can partition rows
// across their existing workers by passing row-sliced A and C. ARM64 SIMD uses
// FP32 fused multiply-add; results need not be bit-identical to scalar builds
// or other BLAS reduction orders. See README.md for the numerical contract.
func (b *PackedB) Mul(dst []float32, dstStride int, a []float32, aStride int, m int) error {
	if b == nil {
		return ErrNilMatrix
	}
	if !validMatrix(a, m, b.k, aStride) || !validMatrix(dst, m, b.n, dstStride) {
		return ErrShape
	}
	b.mul(dst, dstStride, a, aStride, m)
	return nil
}

// mul requires validated shapes and exclusive ownership of the output rows.
func (b *PackedB) mul(dst []float32, dstStride int, a []float32, aStride int, m int) {
	if m == 0 || b.n == 0 {
		return
	}
	if b.k == 0 {
		for r := 0; r < m; r++ {
			clear(dst[r*dstStride : r*dstStride+b.n])
		}
		return
	}
	mulPacked(dst, dstStride, a, aStride, b.data, m, b.k, b.n)
}

func validMatrix(values []float32, rows, cols, stride int) bool {
	if rows < 0 || cols < 0 || stride < 0 {
		return false
	}
	if rows == 0 || cols == 0 {
		return true
	}
	if stride < cols || len(values) < cols {
		return false
	}
	// Division avoids overflow for hostile dimensions and strides.
	return rows-1 <= (len(values)-cols)/stride
}
