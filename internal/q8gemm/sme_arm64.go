// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64

package q8gemm

import "sync/atomic"

//go:noescape
func smeMul16x64(q *int8, k, panels int, activation *float32, dst *float32, n, rows int) (retries int)

//go:noescape
func smeVectorBytes() int

var (
	smeEnabled = smeSupported() && smeVectorBytes() == 64
	smeRetries atomic.Uint64
)
