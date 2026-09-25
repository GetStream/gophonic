// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64

package whisper

import (
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/nn"
)

func TestNormLargeOffsetRetainsVariance(t *testing.T) {
	src := []float32{1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6 + 0.0625}
	gamma := []float32{1, 1, 1, 1, 1, 1, 1, 1}
	beta := make([]float32, len(src))
	want := make([]float32, len(src))
	layerNormReference(src, want, gamma, beta)
	for _, tc := range []struct {
		name string
		run  func([]float32)
	}{
		{"layerNormRow", func(dst []float32) { nn.LayerNorm(src, dst, gamma, beta) }},
		{"residualNormInto", func(dst []float32) {
			row := append([]float32(nil), src...)
			residualNormInto(row, dst, make([]float32, len(src)), gamma, beta)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]float32, len(src))
			tc.run(got)
			for i := range got {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 1e-3 {
					t.Fatalf("output[%d]=%g, want %g", i, got[i], want[i])
				}
			}
		})
	}
}
