// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

// runTinyConvScalarRange is the portable range kernel used by non-SIMD builds
// and by SIMD cases that do not have packed dense weights. fromT and toT are
// half-open output frame indexes.
func runTinyConvScalarRange(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength, fromT, toT int) int {
	pad := conv.kernel / 2
	outputLength := (inputLength+2*pad-conv.kernel)/conv.stride + 1
	if fromT < 0 {
		fromT = 0
	}
	if toT > outputLength {
		toT = outputLength
	}
	inPerGroup := conv.inChannels / conv.groups
	outPerGroup := conv.outChannels / conv.groups
	scale := params.scale * conv.weightScale
	for t := fromT; t < toT; t++ {
		for oc := 0; oc < conv.outChannels; oc++ {
			group := oc / outPerGroup
			var accumulator int32
			for icg := 0; icg < inPerGroup; icg++ {
				ic := group*inPerGroup + icg
				weightBase := (oc*inPerGroup + icg) * conv.kernel
				for k := 0; k < conv.kernel; k++ {
					it := t*conv.stride + k - pad
					if it < 0 || it >= inputLength {
						continue
					}
					x := int32(input[it*conv.inChannels+ic]) - int32(params.zero)
					w := int32(conv.weight[weightBase+k]) - int32(conv.weightZero)
					accumulator += x * w
				}
			}
			value := float32(accumulator) * scale
			output[t*conv.outChannels+oc] = value + conv.bias[oc]
		}
	}
	return outputLength
}
