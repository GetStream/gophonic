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
