// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor_test

import (
	"errors"
	"testing"

	"github.com/GetStream/gofloor"
)

// exampleExternalSession models how a separately implemented Go architecture
// can satisfy the public session contract and reuse only the shared frontend.
type exampleExternalSession struct {
	frontend *gofloor.WhisperFeatureWorkspace
	features []float32
	closed   bool
}

var _ gofloor.AudioSession = (*exampleExternalSession)(nil)

func newExampleExternalSession() *exampleExternalSession {
	return &exampleExternalSession{
		frontend: gofloor.NewWhisperFeatureWorkspace(),
		features: make([]float32, 80*800),
	}
}

func (s *exampleExternalSession) PredictInto(pcm []float32, sampleRate, channels int) (gofloor.Prediction, error) {
	if s == nil || s.closed {
		return gofloor.Prediction{}, gofloor.ErrSessionClosed
	}
	if err := gofloor.ExtractWhisperFeaturesInto(pcm, sampleRate, channels, s.features, s.frontend); err != nil {
		return gofloor.Prediction{}, err
	}
	// A test-only stand-in for a third-party model head. The interface makes no
	// assumptions about a backend's model representation or inference engine.
	return gofloor.Prediction{Probability: 0.25}, nil
}

func (s *exampleExternalSession) Close() error {
	if s != nil && !s.closed {
		s.closed = true
		s.frontend.Close()
	}
	return nil
}

func TestExternalPackageCanImplementAudioSession(t *testing.T) {
	pcm := make([]float32, 16000)
	custom := newExampleExternalSession()
	defer custom.Close()
	var session gofloor.AudioSession = custom

	got, err := session.PredictInto(pcm, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != (gofloor.Prediction{Probability: 0.25}) {
		t.Fatalf("external session prediction = %+v", got)
	}
	if allocs := testing.AllocsPerRun(3, func() {
		if _, err := session.PredictInto(pcm, 16000, 1); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Fatalf("external session allocated %.1f objects per warm prediction", allocs)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := session.PredictInto(pcm, 16000, 1); !errors.Is(err, gofloor.ErrSessionClosed) {
		t.Fatalf("prediction after Close error = %v, want ErrSessionClosed", err)
	}
}
