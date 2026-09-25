// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import "github.com/GetStream/gophonic/internal/mel"

// fullFeatureRightPad is the silence OpenAI Whisper appends to a recording.
const fullFeatureRightPad = featureSamples

var (
	errNegativeFullFeatureSamples = mel.ErrNegativeSamples
	errFullFeatureSizeOverflow    = mel.ErrTooLarge
	errFullFeatureOutputSize      = mel.ErrFeatureBuffer
)

// FullFeatureWorkspace owns reusable scratch for one concurrent full-file
// audio frontend. It must not be used by more than one call at a time.
type FullFeatureWorkspace struct {
	spectrogram *mel.Spectrogram
	closed      bool
}

// NewFullFeatureWorkspace creates reusable scratch for FullFeaturesInto.
// The workspace grows to fit longer audio and retains that capacity for later
// calls. Allocate one workspace for each concurrent caller.
func NewFullFeatureWorkspace() *FullFeatureWorkspace {
	return &FullFeatureWorkspace{spectrogram: mel.NewSpectrogram(MelBins, fullFeatureRightPad)}
}

// Close releases the workspace buffers. Calls after Close return an error.
// Close must not race with FullFeaturesInto.
func (w *FullFeatureWorkspace) Close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	w.spectrogram = nil
}

// FullFeatureFrames returns the number of Whisper log-mel frames produced for
// samples mono float32 PCM values sampled at 16 kHz. OpenAI Whisper appends
// thirty seconds of silence before its centered STFT and drops the final STFT
// frame, so the result is 3000 + floor(samples/160).
func FullFeatureFrames(samples int) (int, error) {
	return mel.SpectrogramFrames(samples, fullFeatureRightPad, MelBins)
}

// FullFeaturesInto computes OpenAI Whisper's 80-band log-mel spectrogram for
// the entire mono float32 PCM input sampled at 16 kHz. It appends thirty
// seconds of silence, uses a centered 400-point STFT with a 160-sample hop,
// drops the final STFT frame, and applies max-minus-eight normalization over
// the complete spectrogram. dst is channel-major [80, frames], where frames is
// returned by FullFeatureFrames(len(pcm)).
//
// The frontend follows OpenAI Whisper source commit
// 86098128c0b4f24f0e2aa2994de830614b474227. Once the workspace has grown for
// the input size, calls with the same or shorter audio allocate no memory.
// The workspace must be private to this call lane.
func FullFeaturesInto(pcm []float32, dst []float32, w *FullFeatureWorkspace) error {
	if w == nil || w.closed {
		return errNilFeatureWorkspace
	}
	return w.spectrogram.Into(pcm, dst)
}
