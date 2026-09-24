// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package q8gemm

func usingSME() bool { return false }

var _ = forcePortable

func mulPanelsSME([]float32, int, *Workspace, *Weights, int, int) bool { return false }

func retryCount() uint64 { return 0 }

func mulPanelsI8SME([]float32, int, *WorkspaceI8, *WeightsI8, int, int) bool { return false }

func mulRowI8SME([]float32, *WorkspaceI8, *WeightsI8, int, int) bool { return false }

func mulRowF16SME([]float32, *Workspace, *Weights, int, int) bool { return false }
