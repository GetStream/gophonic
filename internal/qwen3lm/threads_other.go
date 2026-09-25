// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !darwin

package qwen3lm

// PerformanceCores reports 0: this platform does not distinguish core types.
func PerformanceCores() int { return 0 }
