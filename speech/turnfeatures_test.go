// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech_test

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/mel"
	"github.com/GetStream/gophonic/speech"
)

func TestTurnFeaturesParity(t *testing.T) {
	pcm := readFloatFixture(t, "../testdata/tone.pcm.f32le")
	w := speech.NewTurnFeatures()
	defer w.Close()
	got := make([]float32, mel.TurnBins*mel.TurnFrames)
	if err := w.Into(pcm, 16000, 1, got); err != nil {
		t.Fatal(err)
	}
	want := readFloatFixture(t, "../testdata/tone.mel.f32le")
	maxError := 0.0
	for i := range got {
		error := math.Abs(float64(got[i] - want[i]))
		if error > maxError {
			maxError = error
		}
	}
	if maxError > 2e-5 {
		t.Fatalf("Whisper oracle max error %g exceeds 2e-5", maxError)
	}

	// A non-16 kHz stereo path must match the frontend used directly, bit for bit.
	stereo := make([]float32, 48000*2)
	for i := 0; i < len(stereo)/2; i++ {
		stereo[2*i] = float32(math.Sin(float64(i) * 0.017))
		stereo[2*i+1] = float32(math.Cos(float64(i) * 0.023))
	}
	old := mel.NewTurn()
	oldFeatures := make([]float32, mel.TurnBins*mel.TurnFrames)
	if err := old.Load(stereo, 48000, 2); err != nil {
		t.Fatal(err)
	}
	if err := old.FeaturesInto(oldFeatures); err != nil {
		t.Fatal(err)
	}
	if err := w.Into(stereo, 48000, 2, got); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(oldFeatures[i]) {
			t.Fatalf("48 kHz stereo feature[%d] differs: %g vs %g", i, got[i], oldFeatures[i])
		}
	}
}

func TestTurnFeaturesWarmZeroAllocAndClose(t *testing.T) {
	pcm := readFloatFixture(t, "../testdata/tone.pcm.f32le")
	w := speech.NewTurnFeatures()
	dst := make([]float32, mel.TurnBins*mel.TurnFrames)
	if err := w.Into(pcm, 16000, 1, dst); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(10, func() {
		if err := w.Into(pcm, 16000, 1, dst); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm Whisper extraction allocated %.1f objects", allocs)
	}
	w.Close()
	w.Close()
	if err := w.Into(pcm, 16000, 1, dst); err == nil {
		t.Fatalf("closed workspace error = %v, want an error", err)
	}
	var none *speech.TurnFeatures
	if err := none.Into(pcm, 16000, 1, dst); err == nil {
		t.Fatalf("nil frontend error = %v, want an error", err)
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

// exampleExternalSession models how a separately implemented Go architecture
// can satisfy the public detector contract and reuse only the shared frontend.
type exampleExternalSession struct {
	frontend *speech.TurnFeatures
	features []float32
	closed   bool
}

var _ speech.TurnDetector = (*exampleExternalSession)(nil)

func newExampleExternalSession() *exampleExternalSession {
	return &exampleExternalSession{
		frontend: speech.NewTurnFeatures(),
		features: make([]float32, 80*800),
	}
}

func (s *exampleExternalSession) Predict(pcm []float32, sampleRate, channels int) (speech.Prediction, error) {
	if s == nil || s.closed {
		return speech.Prediction{}, speech.ErrClosed
	}
	if err := s.frontend.Into(pcm, sampleRate, channels, s.features); err != nil {
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
