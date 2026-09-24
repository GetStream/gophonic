// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

// lowerChannelMajor3 gathers the three-tap, stride-one padded convolution
// input into contiguous rows. The columns retain PyTorch's [channel,kernel]
// weight order, allowing the same packed GEMM as the transformer projections.
func lowerChannelMajor3(src, dst []float32, frames, channels int) {
	for t := 0; t < frames; t++ {
		row := dst[t*channels*3 : (t+1)*channels*3]
		for channel := 0; channel < channels; channel++ {
			in := src[channel*frames : (channel+1)*frames]
			left, right := float32(0), float32(0)
			if t > 0 {
				left = in[t-1]
			}
			if t+1 < frames {
				right = in[t+1]
			}
			row[channel*3] = left
			row[channel*3+1] = in[t]
			row[channel*3+2] = right
		}
	}
}

// lowerTimeMajor3Stride2 gathers the second stem convolution's time-major
// input. Zero padding is written on every call, including after scratch reuse.
func lowerTimeMajor3Stride2(src, dst []float32, inputFrames, outputFrames, channels int) {
	for t := 0; t < outputFrames; t++ {
		center := t * 2
		row := dst[t*channels*3 : (t+1)*channels*3]
		for channel := 0; channel < channels; channel++ {
			left, right := float32(0), float32(0)
			if center > 0 {
				left = src[(center-1)*channels+channel]
			}
			if center+1 < inputFrames {
				right = src[(center+1)*channels+channel]
			}
			row[channel*3] = left
			row[channel*3+1] = src[center*channels+channel]
			row[channel*3+2] = right
		}
	}
}
