// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package nn

import "math"

// LayerNorm writes LayerNorm(src)·gamma+beta to dst with epsilon 1e-5 and
// float64 statistics. Rows whose length is a positive multiple of eight use
// the NEON kernel on arm64, which sums in four float64x2 partial
// accumulators instead of one.
func LayerNorm(src, dst, gamma, beta []float32) {
	n := len(src)
	if Accelerated && n > 0 && n%8 == 0 && len(dst) >= n && len(gamma) >= n && len(beta) >= n {
		layerNormNEON(&src[0], &dst[0], &gamma[0], &beta[0], n)
		return
	}
	layerNormGeneric(src, dst, gamma, beta)
}

func layerNormGeneric(src, dst, gamma, beta []float32) {
	var sum float64
	for _, x := range src {
		sum += float64(x)
	}
	mean := sum / float64(len(src))
	var variance float64
	for _, x := range src {
		delta := float64(x) - mean
		variance += delta * delta
	}
	variance /= float64(len(src))
	invStd := 1 / math.Sqrt(variance+1e-5)
	for i, x := range src {
		normalized := (float64(x) - mean) * invStd
		dst[i] = float32(normalized*float64(gamma[i]) + float64(beta[i]))
	}
}

// ResidualNorm adds add (plus bias, when non-nil) to row in place, then
// writes LayerNorm(row) to dst; dst may alias add. The sum
// row + (add + bias) keeps the same order on every path.
func ResidualNorm(row, dst, gamma, beta, add, bias []float32) {
	n := len(row)
	if Accelerated && n > 0 && n%8 == 0 && len(dst) >= n && len(add) >= n && len(gamma) >= n && len(beta) >= n && (bias == nil || len(bias) >= n) {
		var b *float32
		if bias != nil {
			b = &bias[0]
		}
		residualNormNEON(&row[0], &dst[0], &gamma[0], &beta[0], n, &add[0], b)
		return
	}
	if bias != nil {
		for i := range row {
			row[i] += add[i] + bias[i]
		}
	} else {
		for i := range row {
			row[i] += add[i]
		}
	}
	layerNormGeneric(row, dst, gamma, beta)
}
