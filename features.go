// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"errors"

	"github.com/GetStream/gophonic/internal/mel"
)

var errNilWorkspace = errors.New("gophonic: feature workspace is nil or closed")

// WhisperFeatureWorkspace owns the scratch of the turn detectors' shared
// waveform-to-log-mel frontend. It is private to one prediction lane and can
// be reused by turn-detector backends with their own model architecture.
type WhisperFeatureWorkspace struct {
	audio  *mel.Turn
	closed bool
}

// NewWhisperFeatureWorkspace creates reusable frontend scratch without
// allocating any model's inference buffers or worker pool.
func NewWhisperFeatureWorkspace() *WhisperFeatureWorkspace {
	return &WhisperFeatureWorkspace{audio: mel.NewTurn()}
}

// Close marks the workspace unusable. It is safe to call more than once,
// provided no extraction is in progress.
func (w *WhisperFeatureWorkspace) Close() {
	if w != nil {
		w.closed = true
	}
}

// ExtractWhisperFeaturesInto writes normalized row-major [80,800] log-mel
// features from mono or stereo PCM at 8–96 kHz, as Smart Turn and TinyMelNet
// consume them. Audio is right-aligned to the latest eight seconds. The caller
// owns dst and must not use w concurrently. Once a sample rate is warm,
// successful calls do not allocate.
func ExtractWhisperFeaturesInto(pcm []float32, sampleRate, channels int, dst []float32, w *WhisperFeatureWorkspace) error {
	if w == nil || w.closed {
		return errNilWorkspace
	}
	if len(dst) != mel.TurnBins*mel.TurnFrames {
		return mel.ErrFeatureBuffer
	}
	if err := w.audio.Load(pcm, sampleRate, channels); err != nil {
		return err
	}
	return w.audio.FeaturesInto(dst)
}
