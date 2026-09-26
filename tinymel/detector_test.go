// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package tinymel

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
)

func TestDetectorConstructorAndClosedState(t *testing.T) {
	if _, err := NewDetector(nil, LaneOptions{}); !errors.Is(err, ErrNilModel) {
		t.Fatalf("NewDetector(nil, LaneOptions{}) error = %v, want ErrNilModel", err)
	}
	var s *Detector
	if _, err := s.Predict(nil, 16000, 1); !errors.Is(err, speech.ErrClosed) {
		t.Fatalf("nil Detector prediction error = %v, want speech.ErrClosed", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("closing nil Detector: %v", err)
	}
}

func TestDetectorAdapterParityAndZeroAllocations(t *testing.T) {
	path := testmodels.Path(t, testmodels.TinyMel)
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

	concrete, err := NewDetector(model, LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer concrete.Close()
	var detector speech.TurnDetector = concrete
	got, err := detector.Predict(pcm, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("adapter prediction %+v, direct prediction %+v", got, want)
	}
	if allocs := testing.AllocsPerRun(3, func() {
		if _, err := detector.Predict(pcm, 16000, 1); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Fatalf("interface-dispatched TinyMel detector allocated %.1f objects per warm prediction", allocs)
	}

	if err := detector.Close(); err != nil {
		t.Fatal(err)
	}
	if err := detector.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := detector.Predict(pcm, 16000, 1); !errors.Is(err, speech.ErrClosed) {
		t.Fatalf("prediction after Close error = %v, want speech.ErrClosed", err)
	}

	if got.Probability < 0 || got.Probability > 1 || math.IsNaN(float64(got.Probability)) {
		t.Fatalf("TinyMel detector returned invalid probability %v", got.Probability)
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
