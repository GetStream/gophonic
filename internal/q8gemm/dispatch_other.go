// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package q8gemm

func usingSME() bool { return false }

func mulSME([]float32, []float32, int, *Weights) bool { return false }
