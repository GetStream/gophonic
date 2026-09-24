// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package whisper

const geluAccelerated = false

func geluNEON(values, bias *float32, n int, table *[1025][2]float32) { panic("unreachable") }
