// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64 && !darwin

package whispergemm

// Streaming-mode signal behavior is verified only on Darwin.
func smeSupported() bool { return false }
