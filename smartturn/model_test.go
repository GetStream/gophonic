// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package smartturn

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
)

func TestPredictionMatchesOnnxOracle(t *testing.T) {
	modelPath := testmodels.Path(t, testmodels.SmartTurn)
	model, err := Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	ws := NewWorkspace()
	defer ws.Close()
	zeroFeatures := make([]float32, melCount*frameCount)
	check := func(name string, features []float32, want float32) {
		t.Helper()
		got, err := model.PredictFeaturesInto(features, ws)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(got.Probability-want)) > 2e-5 {
			t.Errorf("%s probability = %.9f, want %.9f", name, got.Probability, want)
		}
	}
	check("zero features", zeroFeatures, 0.98987246)
	toneFeatures := readFloatFixture(t, "../testdata/tone.mel.f32le")
	check("440 Hz tone", toneFeatures, 0.0663098394870758)
	patternFeatures := make([]float32, melCount*frameCount)
	for i := range patternFeatures {
		patternFeatures[i] = float32((i*73)%1009-504) / 256
	}
	check("deterministic feature pattern", patternFeatures, 0.97055656)
	allocs := testing.AllocsPerRun(3, func() {
		if _, err := model.PredictFeaturesInto(zeroFeatures, ws); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Errorf("steady-state PredictFeaturesInto allocated %.1f objects per call, want zero", allocs)
	}
}

func TestSessionConstructorAndClosedState(t *testing.T) {
	if _, err := NewSession(nil); !errors.Is(err, ErrNilModel) {
		t.Fatalf("NewSession(nil) error = %v, want ErrNilModel", err)
	}
	var s *Session
	if _, err := s.Predict(nil, 16000, 1); !errors.Is(err, speech.ErrClosed) {
		t.Fatalf("nil Session prediction error = %v, want speech.ErrClosed", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("closing nil Session: %v", err)
	}
}

func TestSessionAdapterParityAndZeroAllocations(t *testing.T) {
	path := testmodels.Path(t, testmodels.SmartTurn)
	model, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	pcm := readFloatFixture(t, "../testdata/tone.pcm.f32le")

	directWorkspace := NewWorkspace()
	defer directWorkspace.Close()
	want, err := model.PredictMono16kInto(pcm, directWorkspace)
	if err != nil {
		t.Fatal(err)
	}

	concrete, err := NewSession(model)
	if err != nil {
		t.Fatal(err)
	}
	defer concrete.Close()
	var session speech.TurnDetector = concrete
	got, err := session.Predict(pcm, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("adapter prediction %+v, direct prediction %+v", got, want)
	}
	if allocs := testing.AllocsPerRun(3, func() {
		if _, err := session.Predict(pcm, 16000, 1); err != nil {
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
	if _, err := session.Predict(pcm, 16000, 1); !errors.Is(err, speech.ErrClosed) {
		t.Fatalf("prediction after Close error = %v, want speech.ErrClosed", err)
	}
}

func readFloatFixture(t testing.TB, path string) []float32 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data)%4 != 0 {
		t.Fatalf("fixture %s has invalid float32 byte length %d", path, len(data))
	}
	values := make([]float32, len(data)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return values
}
