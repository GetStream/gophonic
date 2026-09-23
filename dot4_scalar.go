// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || (!amd64 && !arm64)

package gofloor

func dotProduct4(a, b0, b1, b2, b3 []float32) (float32, float32, float32, float32) {
	var s0, s1, s2, s3 float32
	for i, x := range a {
		s0 += x * b0[i]
		s1 += x * b1[i]
		s2 += x * b2[i]
		s3 += x * b3[i]
	}
	return s0, s1, s2, s3
}
