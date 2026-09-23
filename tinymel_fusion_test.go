// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && (arm64 || amd64)

package gophonic

import (
	"math"
	"runtime"
	"testing"
)

// Odd output lengths exercise empty caller ranges and partial worker chunks.
// The trailing sentinel catches activation beyond this convolution's output.
func TestTinyConvGELUFusionPreservesRanges(t *testing.T) {
	previous := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(previous)
	convs := []tinyConv1D{
		syntheticTinyConv(80, 192, 5, 2, 1, 371),
		syntheticTinyConv(32, 12, 1, 1, 1, 372),
		syntheticTinyConv(19, 7, 3, 2, 1, 373),
		syntheticTinyConv(32, 32, 5, 2, 32, 374),
	}
	params := tinyQuantParams{scale: 0.015625, zero: 131}
	for _, helpers := range []int{0, 1, 3, 7} {
		ws := NewTinyMelWorkspaceWithWorkers(helpers)
		for c, conv := range convs {
			conv.depthwisePacked = packTinyDepthwiseWeights(conv)
			for _, outputLength := range []int{1, 15, 16, 17, 33} {
				inputLength := (outputLength-1)*conv.stride + 1
				input := make([]uint8, inputLength*conv.inChannels)
				for i := range input {
					input[i] = uint8(29*i + 117)
				}
				count := outputLength * conv.outChannels
				want, got := make([]float32, count+13), make([]float32, count+13)
				for i := range want {
					want[i], got[i] = -0.75, -0.75
				}
				runTinyConv(conv, input, params, want, inputLength)
				tinyGELUInPlace(want[:count])
				length := ws.runConvGELU(conv, input, params, got, inputLength)
				if length != outputLength {
					t.Fatalf("helpers=%d conv=%d length=%d, want %d", helpers, c, length, outputLength)
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("helpers=%d conv=%d length=%d output[%d]=%g, want %g", helpers, c, outputLength, i, got[i], want[i])
					}
				}
				if allocs := testing.AllocsPerRun(3, func() {
					ws.runConvGELU(conv, input, params, got, inputLength)
				}); allocs != 0 {
					t.Fatalf("helpers=%d conv=%d fused activation allocated %g objects", helpers, c, allocs)
				}
			}
		}
		ws.Close()
	}
}
