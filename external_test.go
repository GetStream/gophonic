// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic_test

import (
	"errors"
	"testing"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/speech"
)

// exampleExternalSession models how a separately implemented Go architecture
// can satisfy the public session contract and reuse only the shared frontend.
type exampleExternalSession struct {
	frontend *gophonic.WhisperFeatureWorkspace
	features []float32
	closed   bool
}

var _ speech.TurnDetector = (*exampleExternalSession)(nil)

func newExampleExternalSession() *exampleExternalSession {
	return &exampleExternalSession{
		frontend: gophonic.NewWhisperFeatureWorkspace(),
		features: make([]float32, 80*800),
	}
}

func (s *exampleExternalSession) Predict(pcm []float32, sampleRate, channels int) (speech.Prediction, error) {
	if s == nil || s.closed {
		return speech.Prediction{}, speech.ErrClosed
	}
	if err := gophonic.ExtractWhisperFeaturesInto(pcm, sampleRate, channels, s.features, s.frontend); err != nil {
		return speech.Prediction{}, err
	}
	// A test-only stand-in for a third-party model head. The interface makes no
	// assumptions about a backend's model representation or inference engine.
	return speech.Prediction{Probability: 0.25}, nil
}

func (s *exampleExternalSession) Close() error {
	if s != nil && !s.closed {
		s.closed = true
		s.frontend.Close()
	}
	return nil
}

func TestExternalPackageCanImplementTurnDetector(t *testing.T) {
	pcm := make([]float32, 16000)
	custom := newExampleExternalSession()
	defer custom.Close()
	var session speech.TurnDetector = custom

	got, err := session.Predict(pcm, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != (speech.Prediction{Probability: 0.25}) {
		t.Fatalf("external session prediction = %+v", got)
	}
	if allocs := testing.AllocsPerRun(3, func() {
		if _, err := session.Predict(pcm, 16000, 1); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Fatalf("external session allocated %.1f objects per warm prediction", allocs)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := session.Predict(pcm, 16000, 1); !errors.Is(err, speech.ErrClosed) {
		t.Fatalf("prediction after Close error = %v, want speech.ErrClosed", err)
	}
}
