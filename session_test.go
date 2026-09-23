// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor

import (
	"errors"
	"math"
	"os"
	"testing"
)

func TestAudioSessionConstructorsAndClosedState(t *testing.T) {
	if _, err := NewSmartTurnSession(nil); !errors.Is(err, ErrNilModel) {
		t.Fatalf("NewSmartTurnSession(nil) error = %v, want ErrNilModel", err)
	}
	if _, err := NewTinyMelSession(nil, 0); !errors.Is(err, ErrNilModel) {
		t.Fatalf("NewTinyMelSession(nil, 0) error = %v, want ErrNilModel", err)
	}

	var smart *SmartTurnSession
	if _, err := smart.PredictInto(nil, 16000, 1); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("nil SmartTurnSession prediction error = %v, want ErrSessionClosed", err)
	}
	if err := smart.Close(); err != nil {
		t.Fatalf("closing nil SmartTurnSession: %v", err)
	}

	var tiny *TinyMelSession
	if _, err := tiny.PredictInto(nil, 16000, 1); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("nil TinyMelSession prediction error = %v, want ErrSessionClosed", err)
	}
	if err := tiny.Close(); err != nil {
		t.Fatalf("closing nil TinyMelSession: %v", err)
	}
}

func TestSmartTurnSessionAdapterParityAndZeroAllocations(t *testing.T) {
	path := os.Getenv("GOFLOOR_TEST_MODEL")
	if path == "" {
		t.Skip("set GOFLOOR_TEST_MODEL to a converted Smart Turn bundle to run session adapter checks")
	}
	model, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")

	directWorkspace := NewWorkspace()
	defer directWorkspace.Close()
	want, err := model.PredictMono16kInto(pcm, directWorkspace)
	if err != nil {
		t.Fatal(err)
	}

	concrete, err := NewSmartTurnSession(model)
	if err != nil {
		t.Fatal(err)
	}
	defer concrete.Close()
	var session AudioSession = concrete
	got, err := session.PredictInto(pcm, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("adapter prediction %+v, direct prediction %+v", got, want)
	}
	if allocs := testing.AllocsPerRun(3, func() {
		if _, err := session.PredictInto(pcm, 16000, 1); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Fatalf("interface-dispatched Smart Turn session allocated %.1f objects per warm prediction", allocs)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := session.PredictInto(pcm, 16000, 1); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("prediction after Close error = %v, want ErrSessionClosed", err)
	}
}

func TestTinyMelSessionAdapterParityAndZeroAllocations(t *testing.T) {
	path := os.Getenv("GOFLOOR_TEST_TINYMEL_MODEL")
	if path == "" {
		t.Skip("set GOFLOOR_TEST_TINYMEL_MODEL to a converted TinyMelNet bundle to run session adapter checks")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		t.Fatal(err)
	}
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")

	directWorkspace := NewTinyMelWorkspace()
	defer directWorkspace.Close()
	want, err := model.PredictMono16kInto(pcm, directWorkspace)
	if err != nil {
		t.Fatal(err)
	}

	concrete, err := NewTinyMelSession(model, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer concrete.Close()
	var session AudioSession = concrete
	got, err := session.PredictInto(pcm, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("adapter prediction %+v, direct prediction %+v", got, want)
	}
	if allocs := testing.AllocsPerRun(3, func() {
		if _, err := session.PredictInto(pcm, 16000, 1); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Fatalf("interface-dispatched TinyMel session allocated %.1f objects per warm prediction", allocs)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := session.PredictInto(pcm, 16000, 1); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("prediction after Close error = %v, want ErrSessionClosed", err)
	}

	if got.Probability < 0 || got.Probability > 1 || math.IsNaN(float64(got.Probability)) {
		t.Fatalf("TinyMel session returned invalid probability %v", got.Probability)
	}
}
