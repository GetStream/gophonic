// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package tinymel

import (
	"math"
)

type tinyQuantParams struct {
	scale float32
	zero  uint8
}

// quantizeTiny applies ONNX DynamicQuantizeLinear to a contiguous activation.
// The output uses the asymmetric QUInt8 range [0,255].
func quantizeTinyScalar(input []float32, output []uint8) tinyQuantParams {
	minValue, maxValue := float32(0), float32(0)
	for _, value := range input {
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
	}
	scale := (maxValue - minValue) / 255
	if scale == 0 {
		// ONNX Runtime uses a unit scale for an all-zero tensor.
		scale = 1
	}
	zeroFloat := float32(math.RoundToEven(float64(-minValue / scale)))
	if zeroFloat < 0 {
		zeroFloat = 0
	} else if zeroFloat > 255 {
		zeroFloat = 255
	}
	zero := uint8(zeroFloat)
	for i, value := range input {
		q := float32(math.RoundToEven(float64(value/scale))) + float32(zero)
		if q < 0 {
			q = 0
		} else if q > 255 {
			q = 255
		}
		output[i] = uint8(q)
	}
	return tinyQuantParams{scale: scale, zero: zero}
}

// quantizeTinyMel applies one DynamicQuantizeLinear range over the complete
// band-major mel input while writing channels-last bytes for the convolutions.
func quantizeTinyMel(mel []float32, output []uint8) tinyQuantParams {
	minValue, maxValue := float32(0), float32(0)
	for _, value := range mel {
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
	}
	scale := (maxValue - minValue) / 255
	if scale == 0 {
		scale = 1
	}
	zeroFloat := float32(math.RoundToEven(float64(-minValue / scale)))
	if zeroFloat < 0 {
		zeroFloat = 0
	} else if zeroFloat > 255 {
		zeroFloat = 255
	}
	zero := uint8(zeroFloat)
	for t := 0; t < tinyFrameCount; t++ {
		for c := 0; c < tinyMelCount; c++ {
			value := mel[c*tinyFrameCount+t]
			q := float32(math.RoundToEven(float64(value/scale))) + float32(zero)
			if q < 0 {
				q = 0
			} else if q > 255 {
				q = 255
			}
			output[t*tinyMelCount+c] = uint8(q)
		}
	}
	return tinyQuantParams{scale: scale, zero: zero}
}
