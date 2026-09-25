// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64

package q8gemm

// packPairsNEON scales pairs*2 FP32 values from src, rounds them to FP16 with
// the default round-to-nearest-even mode, and stores pair i at dst+64*i bytes.
// pairs must be a multiple of four.
//
//go:noescape
func packPairsNEON(dst *uint16, src *float32, pairs int, scale float32)

// packQuadsNEON scales quads*4 FP32 values from src, rounds to nearest
// (ties to even), saturates to int8, and stores quad i at dst+64*i bytes.
// quads must be a multiple of four.
//
//go:noescape
func packQuadsNEON(dst *int8, src *float32, quads int, scale float32)

func packRowQuads(dst []int8, src []float32, quads int, scale float32) int {
	vector := quads &^ 3
	if vector > 0 {
		_ = dst[(vector-1)*4*ActivationRows+3]
		_ = src[4*vector-1]
		packQuadsNEON(&dst[0], &src[0], vector, scale)
	}
	return vector
}

// stripI8NEON computes up to 8 rows by 8 columns of the int8 product with
// NEON SDOT; see smesrc/strip8x8i8.S.
//
//go:noescape
func stripI8NEON(w *int8, quads int, act *int8, dst *float32, strideBytes, rows int, colScales, rowScales *float32)

// stripI8 computes 16 columns starting at col for every packed row.
func stripI8(dst []float32, stride int, ws *WorkspaceI8, w *WeightsI8, col int) bool {
	panel, within := col/OutputPanel, col%OutputPanel
	base := panel*w.quads*4*OutputPanel + within*4
	for half := 0; half < StripColumns; half += 8 {
		for r0 := 0; r0 < ws.rows; r0 += 8 {
			stripI8NEON(&w.q[base+half*4], w.quads, &ws.activation[r0*4], &dst[r0*stride+col+half],
				4*stride, min(8, ws.rows-r0), &w.scales[col+half], &ws.rowScale[r0])
		}
	}
	return true
}

// maxAbsNEON returns the largest magnitude of n values; n must be a multiple
// of 16. A NaN input yields NaN.
//
//go:noescape
func maxAbsNEON(src *float32, n int) float32

func packRowPairs(dst []uint16, src []float32, pairs int, scale float32) int {
	vector := pairs &^ 3
	if vector > 0 {
		_ = dst[(vector-1)*2*ActivationRows+1]
		_ = src[2*vector-1]
		packPairsNEON(&dst[0], &src[0], vector, scale)
	}
	return vector
}

func maxAbsPrefix(x []float32) (float32, int) {
	n := len(x) &^ 15
	if n == 0 {
		return 0, 0
	}
	return maxAbsNEON(&x[0], n), n
}

//go:noescape
func packContiguousF16NEON(dst *uint16, src *float32, n int, scale float32)

func packContiguousRow(dst []uint16, src []float32, scale float32) int {
	n := len(src) &^ 7
	if n > 0 {
		_ = dst[n-1]
		packContiguousF16NEON(&dst[0], &src[0], n, scale)
	}
	return n
}
