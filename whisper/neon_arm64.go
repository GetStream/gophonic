// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

// attnPrepNEON applies q = (q+qbias)*scale, k *= scale, and v += vbias to n
// values of one row each; n must be a positive multiple of four.
//
//go:noescape
func attnPrepNEON(q, k, v, qbias, vbias *float32, n int, scale float32)

// maxNumNEON returns the largest non-NaN value of x[:n], or NaN when every
// value is NaN. n must be a positive multiple of sixteen.
//
//go:noescape
func maxNumNEON(x *float32, n int) float32
