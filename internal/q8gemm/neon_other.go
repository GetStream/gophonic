// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package q8gemm

func packRowPairs([]uint16, []float32, int, float32) int { return 0 }

func maxAbsPrefix([]float32) (float32, int) { return 0, 0 }

func packRowQuads([]int8, []float32, int, float32) int { return 0 }
