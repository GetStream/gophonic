// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package smartturn

import (
	"errors"
	"fmt"

	"github.com/GetStream/gophonic/speech"
)

// ErrNilModel is returned when NewSession receives a nil model.
var ErrNilModel = errors.New("smartturn: model is nil")

// Session adapts a Model to speech.TurnDetector. The model is immutable and
// can be shared across sessions; each session owns one reusable Workspace.
type Session struct {
	model     *Model
	workspace *Workspace
	closed    bool
}

var (
	_ speech.TurnDetector    = (*Session)(nil)
	_ speech.AudioClassifier = (*Session)(nil)
)

// turnLabels are ClassifyInto's classes: the turn goes on, or it is over.
var turnLabels = []string{"incomplete", "complete"}

// Labels implements speech.AudioClassifier; the slice must not be modified.
func (s *Session) Labels() []string { return turnLabels }

// ClassifyInto implements speech.AudioClassifier: probs receives the
// probabilities that the turn is incomplete and complete.
func (s *Session) ClassifyInto(pcm []float32, sampleRate, channels int, probs []float32) error {
	if len(probs) != len(turnLabels) {
		return fmt.Errorf("%s: %d probabilities for %d labels", "smartturn", len(probs), len(turnLabels))
	}
	p, err := s.PredictInto(pcm, sampleRate, channels)
	if err != nil {
		return err
	}
	probs[0], probs[1] = 1-p.Probability, p.Probability
	return nil
}

// NewSession creates a Smart Turn session with its own workspace.
func NewSession(model *Model) (*Session, error) {
	if model == nil {
		return nil, ErrNilModel
	}
	return &Session{model: model, workspace: NewWorkspace()}, nil
}

// PredictInto extracts features from interleaved PCM and runs Smart Turn. A
// session may be used by only one goroutine at a time.
func (s *Session) PredictInto(pcm []float32, sampleRate, channels int) (Prediction, error) {
	if s == nil || s.closed {
		return Prediction{}, speech.ErrClosed
	}
	return s.model.PredictInto(pcm, sampleRate, channels, s.workspace)
}

// Close stops the session's worker goroutines. It is safe to call Close more
// than once; a closed session cannot be reused.
func (s *Session) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	s.workspace.Close()
	return nil
}
