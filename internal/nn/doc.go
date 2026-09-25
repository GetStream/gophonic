// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package nn holds the row kernels gophonic's transformer encoders share:
// the exact-erf GELU, LayerNorm with float64 statistics, LayerNorm fused with
// a residual add, and the softmax exponential. On arm64 each has a NEON
// kernel assembled from asmsrc by internal/whispergemm/smesrc/gen.py; other
// platforms run portable code with the same operation order where noted.
package nn
