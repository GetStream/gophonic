// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package tinymel

import (
	"errors"

	"github.com/GetStream/gophonic/speech"
)

// ErrNilModel is returned when NewSession receives a nil model.
var ErrNilModel = errors.New("tinymel: model is nil")

// Session adapts a TinyMelNet Model to speech.TurnDetector. Its helper count
// controls the persistent worker pool owned by the session's workspace.
type Session struct {
	model     *Model
	workspace *Workspace
	closed    bool
}

var _ speech.TurnDetector = (*Session)(nil)

// NewSession creates a TinyMelNet session. helpers is the requested
// persistent helper count; it is clamped to [0, MaxWorkers] and
// GOMAXPROCS-1, matching NewWorkspaceWithWorkers.
func NewSession(model *Model, helpers int) (*Session, error) {
	if model == nil {
		return nil, ErrNilModel
	}
	return &Session{model: model, workspace: NewWorkspaceWithWorkers(helpers)}, nil
}

// PredictInto extracts features from interleaved PCM and runs TinyMelNet. A
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
