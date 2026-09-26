// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech

import (
	"errors"

	"github.com/GetStream/gophonic/internal/mel"
)

// The shape of TurnFeatures: 80 log-mel bands of 800 frames, eight seconds
// at 100 frames a second.
const (
	TurnFeatureBands  = mel.TurnBins
	TurnFeatureFrames = mel.TurnFrames
)

var errTurnFeaturesClosed = errors.New("speech: turn features closed")

// TurnFeatures is the turn detectors' waveform-to-log-mel frontend, as
// Smart Turn and TinyMelNet consume it, for detectors trained on the same
// features: it owns one lane's scratch and no model's.
type TurnFeatures struct {
	audio  *mel.Turn
	closed bool
}

// NewTurnFeatures returns a frontend.
func NewTurnFeatures() *TurnFeatures { return &TurnFeatures{audio: mel.NewTurn()} }

// Into writes normalized, row-major [TurnFeatureBands][TurnFeatureFrames]
// log-mel features of interleaved mono or stereo PCM at 8–96 kHz to dst: the
// latest eight seconds, right-aligned. Once a sample rate is warm, it
// allocates nothing.
func (f *TurnFeatures) Into(pcm []float32, sampleRate, channels int, dst []float32) error {
	if f == nil || f.closed {
		return errTurnFeaturesClosed
	}
	if len(dst) != TurnFeatureBands*TurnFeatureFrames {
		return mel.ErrFeatureBuffer
	}
	if err := f.audio.Load(pcm, sampleRate, channels); err != nil {
		return err
	}
	return f.audio.FeaturesInto(dst)
}

// Close ends the frontend. It is safe to call more than once.
func (f *TurnFeatures) Close() {
	if f != nil {
		f.closed = true
	}
}
