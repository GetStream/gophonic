// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package tinymel

// quantizeTinyMelSIMD uses the contiguous SIMD quantizer for the band-major
// mel tensor, then transposes the quantized bytes into the channels-last
// layout consumed by the stem convolution. scratch must hold len(mel) bytes;
// it is owned by the caller so the steady-state path remains allocation-free.
func quantizeTinyMelSIMD(mel []float32, output, scratch []uint8) tinyQuantParams {
	params := quantizeTinySIMD(mel, scratch)
	transposeTinyMelBytes16(scratch, output)
	return params
}

// transposeTinyMelBytes16 transposes [80,800] band-major bytes into [800,80]
// channels-last bytes in small tiles. Tiling keeps the source rows and output
// rows active in cache together while avoiding a full temporary matrix.
func transposeTinyMelBytes16(input, output []uint8) {
	const tile = 16
	for c0 := 0; c0 < tinyMelCount; c0 += tile {
		for t0 := 0; t0 < tinyFrameCount; t0 += tile {
			for t := t0; t < t0+tile; t++ {
				dst := output[t*tinyMelCount+c0 : t*tinyMelCount+c0+tile]
				for c := 0; c < tile; c++ {
					dst[c] = input[(c0+c)*tinyFrameCount+t]
				}
			}
		}
	}
}

// transposeTinyMelBytesNaive is retained as a benchmark comparison for the
// tiled layout. Each input band is streamed contiguously, with strided writes.
func transposeTinyMelBytesNaive(input, output []uint8) {
	for c := 0; c < tinyMelCount; c++ {
		for t := 0; t < tinyFrameCount; t++ {
			output[t*tinyMelCount+c] = input[c*tinyFrameCount+t]
		}
	}
}

// transposeTinyMelBytesTile is a benchmarkable tile-size variant. The model
// shape is divisible by the tile sizes used in the tests and benchmarks.
func transposeTinyMelBytesTile(input, output []uint8, tileC, tileT int) {
	for c0 := 0; c0 < tinyMelCount; c0 += tileC {
		for t0 := 0; t0 < tinyFrameCount; t0 += tileT {
			cN, tN := tileC, tileT
			if c0+cN > tinyMelCount {
				cN = tinyMelCount - c0
			}
			if t0+tN > tinyFrameCount {
				tN = tinyFrameCount - t0
			}
			for t := t0; t < t0+tN; t++ {
				dst := output[t*tinyMelCount+c0 : t*tinyMelCount+c0+cN]
				for c := 0; c < cN; c++ {
					dst[c] = input[(c0+c)*tinyFrameCount+t]
				}
			}
		}
	}
}
