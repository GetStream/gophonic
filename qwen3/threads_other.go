// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !darwin

package qwen3

func performanceCores() int { return 0 }
