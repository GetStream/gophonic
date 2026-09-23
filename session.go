// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import "errors"

// ErrNilModel is returned when a session constructor receives a nil model.
var ErrNilModel = errors.New("gophonic: model is nil")

// ErrSessionClosed is returned when PredictInto is called after Close.
var ErrSessionClosed = errors.New("gophonic: audio session is closed")

// AudioSession runs one turn detector on interleaved PCM audio.
//
// Implementations own any mutable scratch state needed for predictions. Use a
// separate session for each concurrent prediction lane; PredictInto and Close
// must not overlap on the same session. A custom backend can implement this
// interface without registering itself with gophonic or using an ONNX runtime.
type AudioSession interface {
	PredictInto(pcm []float32, sampleRate, channels int) (Prediction, error)
	Close() error
}

// SmartTurnSession adapts a Smart Turn v3.2 Model to AudioSession. The model is
// immutable and can be shared across sessions; each session owns one reusable
// Workspace.
type SmartTurnSession struct {
	model     *Model
	workspace *Workspace
	closed    bool
}

// NewSmartTurnSession creates a Smart Turn audio session with one reusable
// workspace. Close the session when its prediction lane is no longer needed.
func NewSmartTurnSession(model *Model) (*SmartTurnSession, error) {
	if model == nil {
		return nil, ErrNilModel
	}
	return &SmartTurnSession{model: model, workspace: NewWorkspace()}, nil
}

// PredictInto extracts features from interleaved PCM and runs Smart Turn v3.2.
// A session may be used by only one goroutine at a time.
func (s *SmartTurnSession) PredictInto(pcm []float32, sampleRate, channels int) (Prediction, error) {
	if s == nil || s.closed {
		return Prediction{}, ErrSessionClosed
	}
	return s.model.PredictInto(pcm, sampleRate, channels, s.workspace)
}

// Close stops the session's worker goroutines. It is safe to call Close more
// than once; a closed session cannot be reused.
func (s *SmartTurnSession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	s.workspace.Close()
	return nil
}

// TinyMelSession adapts a TinyMelNet model to AudioSession. Its helper count
// controls the persistent worker pool owned by the session's workspace.
type TinyMelSession struct {
	model     *TinyMelModel
	workspace *TinyMelWorkspace
	closed    bool
}

// NewTinyMelSession creates a TinyMelNet audio session. helpers is the
// requested persistent helper count; it is clamped to [0, MaxTinyMelWorkers]
// and GOMAXPROCS-1, matching NewTinyMelWorkspaceWithWorkers.
func NewTinyMelSession(model *TinyMelModel, helpers int) (*TinyMelSession, error) {
	if model == nil {
		return nil, ErrNilModel
	}
	return &TinyMelSession{model: model, workspace: NewTinyMelWorkspaceWithWorkers(helpers)}, nil
}

// PredictInto extracts features from interleaved PCM and runs TinyMelNet. A
// session may be used by only one goroutine at a time.
func (s *TinyMelSession) PredictInto(pcm []float32, sampleRate, channels int) (Prediction, error) {
	if s == nil || s.closed {
		return Prediction{}, ErrSessionClosed
	}
	return s.model.PredictInto(pcm, sampleRate, channels, s.workspace)
}

// Close stops the session's worker goroutines. It is safe to call Close more
// than once; a closed session cannot be reused.
func (s *TinyMelSession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	s.workspace.Close()
	return nil
}

var (
	_ AudioSession = (*SmartTurnSession)(nil)
	_ AudioSession = (*TinyMelSession)(nil)
)
