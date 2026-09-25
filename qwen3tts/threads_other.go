// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !(darwin && arm64)

package qwen3tts

// matrixUnits reports the streaming matrix units; zero means unknown.
func matrixUnits() int { return 0 }
