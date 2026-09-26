// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package smartturn

import (
	"errors"
	"fmt"

	"github.com/GetStream/gophonic/speech"
)

// ErrNilModel is returned when NewDetector receives a nil model.
var ErrNilModel = errors.New("smartturn: model is nil")

// Detector adapts a Model to speech.TurnDetector. The model is immutable and
// can be shared across detectors; each detector owns one reusable Workspace.
type Detector struct {
	model     *Model
	workspace *Workspace
	closed    bool
}

var (
	_ speech.TurnDetector    = (*Detector)(nil)
	_ speech.AudioClassifier = (*Detector)(nil)
)

// turnLabels are ClassifyInto's classes: the turn goes on, or it is over.
var turnLabels = []string{"incomplete", "complete"}

// Labels implements speech.AudioClassifier; the slice must not be modified.
func (s *Detector) Labels() []string { return turnLabels }

// ClassifyInto implements speech.AudioClassifier: probs receives the
// probabilities that the turn is incomplete and complete.
func (s *Detector) ClassifyInto(pcm []float32, sampleRate, channels int, probs []float32) error {
	if len(probs) != len(turnLabels) {
		return fmt.Errorf("%s: %d probabilities for %d labels", "smartturn", len(probs), len(turnLabels))
	}
	p, err := s.Predict(pcm, sampleRate, channels)
	if err != nil {
		return err
	}
	probs[0], probs[1] = 1-p.Probability, p.Probability
	return nil
}

// NewDetector opens a lane over model, with its own workspace. It runs on
// the calling goroutine.
func NewDetector(model *Model) (*Detector, error) {
	if model == nil {
		return nil, ErrNilModel
	}
	return &Detector{model: model, workspace: NewWorkspace()}, nil
}

// Predict extracts features from interleaved PCM and runs Smart Turn. A
// detector may be used by only one goroutine at a time.
func (s *Detector) Predict(pcm []float32, sampleRate, channels int) (Prediction, error) {
	if s == nil || s.closed {
		return Prediction{}, speech.ErrClosed
	}
	return s.model.PredictInto(pcm, sampleRate, channels, s.workspace)
}

// Close stops the detector's worker goroutines. It is safe to call Close more
// than once; a closed detector cannot be reused.
func (s *Detector) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	s.workspace.Close()
	return nil
}
