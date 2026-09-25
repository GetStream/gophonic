// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package nn

const geluAccelerated = true

// geluNEON replaces values[:n] with GELU(values+bias), where bias may be nil.
// n must be a positive multiple of four. It evaluates the geluNormalTable
// expansion with the same operation order as the archsimd kernel; |x| >= 8
// yields x or -0 without the float64 erf fallback, and NaN propagates.
//
//go:noescape
func geluNEON(values, bias *float32, n int, table *[1025][2]float32)
