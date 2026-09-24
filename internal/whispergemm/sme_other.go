// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !arm64

package whispergemm

import "sync/atomic"

var smeRetries atomic.Int64

const smeEnabled = false

func mulSME([]float32, int, []float32, int, []float32, int, int, int) bool { return false }

func mulPackedVectorSME(*PackedVector, []float32, []float32) bool { return false }

func smeTransposePack(*float32, int, int, int, *float32) { panic("unreachable") }
