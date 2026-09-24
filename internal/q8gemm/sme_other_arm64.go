// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64 && !darwin

package q8gemm

// SME signal behavior and feature reporting have only been verified on Darwin.
func smeSupported() bool { return false }
