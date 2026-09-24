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
