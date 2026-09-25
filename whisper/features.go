// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"

	"github.com/GetStream/gophonic/internal/mel"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

const (
	featureSampleRate = 16000
	featureSamples    = mel.WindowSamples
)

var (
	errNilFeatureWorkspace = errors.New("whisper: nil or closed feature workspace")
	errFeatureOutputSize   = mel.ErrFeatureBuffer
	errFeatureNonFinite    = mel.ErrNonFinite
)

// FeatureWorkspace owns reusable scratch for one concurrent audio frontend.
// It must not be used by more than one call at a time.
type FeatureWorkspace struct {
	window *mel.Window
	closed bool
}

// NewFeatureWorkspace allocates scratch for allocation-free FeaturesInto calls.
// It uses about 4 MiB; allocate one workspace for each concurrent caller.
func NewFeatureWorkspace() *FeatureWorkspace {
	return &FeatureWorkspace{window: mel.NewWindow(MelBins)}
}

// Close releases the workspace buffers. Calls after Close return an error.
// Close must not race with FeaturesInto.
func (w *FeatureWorkspace) Close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	w.window = nil
}

// FeaturesInto computes OpenAI Whisper's 80-band, 3000-frame log-mel input.
// pcm must be mono float32 PCM sampled at 16 kHz. Samples after 30 seconds are
// ignored; shorter inputs are zero-padded on the right. No waveform
// normalization is applied. dst is channel-major [80,3000] and must have
// exactly MelBins*MelFrames elements. Once w is created, calls allocate no
// memory. The workspace must be private to this call lane.
//
// The transform follows OpenAI Whisper source commit
// 86098128c0b4f24f0e2aa2994de830614b474227: periodic Hann window, centered
// reflect padding, power spectrum, Whisper's 80-band Slaney mel bank,
// dropped final STFT frame, 1e-10 log floor, max-minus-8 dynamic floor, and
// (log10(mel)+4)/4 output scaling.
func FeaturesInto(pcm []float32, dst []float32, w *FeatureWorkspace) error {
	return featuresInto(pcm, dst, w, nil)
}

func featuresInto(pcm []float32, dst []float32, w *FeatureWorkspace, executor *whispergemm.Executor) error {
	if w == nil || w.closed {
		return errNilFeatureWorkspace
	}
	return w.window.Into(pcm, dst, executor)
}
