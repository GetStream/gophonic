// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import "math/rand"

func syntheticTinyConv(inChannels, outChannels, kernel, stride, groups int, seed int64) tinyConv1D {
	rng := rand.New(rand.NewSource(seed))
	inPerGroup := inChannels / groups
	weight := make([]uint8, outChannels*inPerGroup*kernel)
	for i := range weight {
		weight[i] = uint8(rng.Intn(256))
	}
	packed := make([]uint8, len(weight))
	for oc := 0; oc < outChannels; oc++ {
		for ic := 0; ic < inPerGroup; ic++ {
			for k := 0; k < kernel; k++ {
				src := (oc*inPerGroup+ic)*kernel + k
				dst := (oc*kernel+k)*inPerGroup + ic
				packed[dst] = weight[src]
			}
		}
	}
	bias := make([]float32, outChannels)
	for i := range bias {
		bias[i] = float32(rng.Intn(100)-50) * 0.0078125
	}
	return tinyConv1D{
		weight: weight, packed: packed, weightScale: 0.0048828125,
		weightZero: 137, bias: bias,
		inChannels: inChannels, outChannels: outChannels,
		kernel: kernel, stride: stride, groups: groups,
	}
}
