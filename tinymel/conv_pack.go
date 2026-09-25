// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package tinymel

// packTinyDepthwiseWeights stores one kernel tap's channels contiguously.
// Packing is performed once with immutable model weights, never per inference.
func packTinyDepthwiseWeights(conv tinyConv1D) []uint8 {
	if conv.groups != conv.inChannels || conv.outChannels != conv.inChannels {
		return nil
	}
	packed := make([]uint8, len(conv.weight))
	for k := 0; k < conv.kernel; k++ {
		for c := 0; c < conv.inChannels; c++ {
			packed[k*conv.inChannels+c] = conv.weight[c*conv.kernel+k]
		}
	}
	return packed
}
