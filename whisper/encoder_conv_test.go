// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

func TestEncoderConvolutionLowering(t *testing.T) {
	want := []float32{
		0, 1, 2, 0, 6, 7,
		1, 2, 3, 6, 7, 8,
		2, 3, 4, 7, 8, 9,
		3, 4, 5, 8, 9, 10,
		4, 5, 0, 9, 10, 0,
	}
	got := make([]float32, len(want))
	for i := range got {
		got[i] = 123 // Padding must overwrite stale scratch values.
	}
	lowerChannelMajor3([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, got, 5, 2)
	if !slices.Equal(got, want) {
		t.Fatalf("channel-major lowering = %v, want %v", got, want)
	}
	strided := make([]float32, 3*2*3)
	for i := range strided {
		strided[i] = 123
	}
	lowerTimeMajor3Stride2([]float32{1, 6, 2, 7, 3, 8, 4, 9, 5, 10}, strided, 5, 3, 2)
	for row := 0; row < 3; row++ {
		if !slices.Equal(strided[row*6:(row+1)*6], want[row*12:row*12+6]) {
			t.Fatalf("stride-two row %d = %v, want %v", row, strided[row*6:(row+1)*6], want[row*12:row*12+6])
		}
	}
}

func TestPackedEncoderConvolutionsMatchDirect(t *testing.T) {
	const frames, inChannels, outChannels = 7, 3, 3
	src := make([]float32, frames*inChannels)
	weights := make([]float32, outChannels*inChannels*3)
	bias := []float32{0.25, -0.5, 1}
	for i := range src {
		src[i] = float32(i%11-5) * 0.25
	}
	for i := range weights {
		weights[i] = float32(i%13-6) * 0.125
	}
	b, err := whispergemm.NewPackedB(inChannels*3, outChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Pack(weights, inChannels*3, true); err != nil {
		t.Fatal(err)
	}
	for _, stride := range []int{1, 2} {
		outputFrames := (frames + stride - 1) / stride
		columns := make([]float32, outputFrames*inChannels*3)
		want, got := make([]float32, outputFrames*outChannels), make([]float32, outputFrames*outChannels)
		if stride == 1 {
			lowerChannelMajor3(src, columns, frames, inChannels)
			conv1DChannelMajor(src, want, weights, bias, frames, inChannels, outChannels)
		} else {
			lowerTimeMajor3Stride2(src, columns, frames, outputFrames, inChannels)
			conv1DStride2TimeMajor(src, want, weights, bias, frames, outputFrames, inChannels)
		}
		if err := b.Mul(got, outChannels, columns, inChannels*3, outputFrames); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			got[i] += bias[i%outChannels]
			if math.Abs(float64(got[i]-want[i])) > 1e-6 {
				t.Fatalf("stride=%d output[%d]=%g, want %g", stride, i, got[i], want[i])
			}
		}
	}
}
