// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

// WhisperFeatureWorkspace owns only the scratch required by the shared
// waveform-to-Whisper-frontend path. It is private to one prediction lane and
// can be reused by turn-detector backends with their own model architecture.
type WhisperFeatureWorkspace struct {
	audio  audioWorkspace
	closed bool
}

// NewWhisperFeatureWorkspace creates reusable frontend scratch without
// allocating either supported model's inference buffers or worker pool.
func NewWhisperFeatureWorkspace() *WhisperFeatureWorkspace {
	return &WhisperFeatureWorkspace{audio: newAudioWorkspace()}
}

// Close marks the workspace unusable. It is safe to call more than once,
// provided no extraction is in progress.
func (w *WhisperFeatureWorkspace) Close() {
	if w != nil {
		w.closed = true
	}
}

// ExtractWhisperFeaturesInto writes normalized row-major [80,800] log-mel
// features from mono or stereo PCM at 8–96 kHz. Audio is right-aligned to the
// latest eight seconds. The caller owns dst and must not use w concurrently.
// Once a sample rate is warm, successful calls do not allocate.
func ExtractWhisperFeaturesInto(pcm []float32, sampleRate, channels int, dst []float32, w *WhisperFeatureWorkspace) error {
	if w == nil || w.closed {
		return errNilWorkspace
	}
	if len(dst) != melCount*frameCount {
		return errInvalidFeatureBuffer
	}
	if err := prepareAudioInto(pcm, sampleRate, channels, &w.audio); err != nil {
		return err
	}
	return computeFeatures16k(w.audio.samples, dst, &w.audio)
}
