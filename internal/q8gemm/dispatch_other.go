// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package q8gemm

func usingSME() bool { return false }

var _ = forcePortable

func mulPanelsSME([]float32, int, *Workspace, *Weights, int, int) bool { return false }

func retryCount() uint64 { return 0 }
