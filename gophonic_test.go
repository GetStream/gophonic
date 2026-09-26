// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
)

func TestOpenRejectsUnknownFormats(t *testing.T) {
	dir := t.TempDir()
	for name, content := range map[string]string{"short": "GO", "other": "NOTAMODEL-------"} {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := gophonic.Open(path, gophonic.Options{}); !errors.Is(err, gophonic.ErrUnknownFormat) {
			t.Errorf("Open(%s) error = %v, want ErrUnknownFormat", name, err)
		}
	}
	if _, err := gophonic.Open(filepath.Join(dir, "missing"), gophonic.Options{}); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("Open(missing) error = %v, want os.ErrNotExist", err)
	}
}

// TestOpenWhisper transcribes through the model-independent interfaces with
// a local tiny.en bundle, as the CLI and HTTP server do.
func TestOpenWhisper(t *testing.T) {
	path := testmodels.Path(t, testmodels.WhisperTinyEN)
	model, err := gophonic.Open(path, gophonic.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "whisper" || !gophonic.Supports[speech.Transcriber](model) {
		t.Fatalf("Open = %s providing %v, want whisper transcription", model.Name(), model.Provides())
	}
	if _, err := gophonic.Lane[speech.TurnDetector](model); !errors.Is(err, speech.ErrUnsupported) {
		t.Fatalf("Lane[TurnDetector] error = %v, want speech.ErrUnsupported", err)
	}
	lane, err := gophonic.Lane[speech.Transcriber](model)
	if err != nil {
		t.Fatal(err)
	}
	defer lane.Close()
	pcm := readFloatFixture(t, "testdata/whisper_jfk.pcm.f32le")
	var out speech.Transcript
	if err := lane.Transcribe(context.Background(), pcm, speech.Options{Words: true}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Text), "ask not what your country can do for you") || out.Language != "English" {
		t.Fatalf("transcript %q (%s)", out.Text, out.Language)
	}
	if len(out.Segments) == 0 || len(out.Words) < 20 {
		t.Fatalf("%d segments, %d words", len(out.Segments), len(out.Words))
	}
	if err := lane.Transcribe(context.Background(), pcm, speech.Options{Language: "de"}, &out); !errors.Is(err, speech.ErrUnsupported) {
		t.Fatalf("German on an English model: error = %v, want speech.ErrUnsupported", err)
	}
}

// A registered format takes over the paths it matches, ahead of the
// built-in ones, and its Model's lanes and Close come from the format.
func TestRegisteredFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.custom")
	if err := os.WriteFile(path, []byte("CUSTOM01"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := gophonic.Open(path, gophonic.Options{}); !errors.Is(err, gophonic.ErrUnknownFormat) {
		t.Fatalf("before Register: %v, want ErrUnknownFormat", err)
	}
	closed := 0
	gophonic.Register(gophonic.Format{
		Name:  "custom",
		Match: func(p string) bool { return strings.HasSuffix(p, ".custom") },
		Open: func(p string, opts gophonic.Options) (*gophonic.Model, error) {
			m := gophonic.NewModel("custom", p, func() error { closed++; return nil })
			gophonic.Provide(m, func() (speech.TurnDetector, error) { return fixedDetector{}, nil })
			// Any interface is a capability, including one gophonic never
			// heard of.
			return gophonic.Provide(m, func() (moderator, error) { return fixedDetector{}, nil }), nil
		},
	})
	if f := gophonic.Formats(); f[0].Name != "custom" || f[len(f)-1].Name != "tinymel" {
		t.Fatalf("formats %v", f)
	}
	model, err := gophonic.Open(path, gophonic.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "custom" || model.Path() != path || len(model.Provides()) != 2 || !gophonic.Supports[moderator](model) {
		t.Fatalf("Open = %s at %s providing %v", model.Name(), model.Path(), model.Provides())
	}
	mod, err := gophonic.Lane[moderator](model)
	if err != nil || !mod.Flag("anything") {
		t.Fatalf("moderator lane: %v", err)
	}
	detector, err := gophonic.Lane[speech.TurnDetector](model)
	if err != nil {
		t.Fatal(err)
	}
	if p, err := detector.Predict(nil, 16000, 1); err != nil || !p.Complete {
		t.Fatalf("Predict = %+v, %v", p, err)
	}
	if _, err := gophonic.Lane[speech.Transcriber](model); !errors.Is(err, speech.ErrUnsupported) {
		t.Fatalf("Lane[Transcriber] error = %v, want speech.ErrUnsupported", err)
	}
	if model.Close() != nil || model.Close() != nil || closed != 1 {
		t.Fatalf("Close ran %d times", closed)
	}
}

// moderator is a capability defined outside gophonic.
type moderator interface{ Flag(text string) bool }

type fixedDetector struct{}

func (fixedDetector) Flag(string) bool { return true }

func (fixedDetector) Predict([]float32, int, int) (speech.Prediction, error) {
	return speech.Prediction{Probability: 1, Complete: true}, nil
}

func (fixedDetector) Close() error { return nil }

// Smart Turn is a turn detector and a binary audio classifier.
func TestOpenSmartTurnClassifies(t *testing.T) {
	model, err := gophonic.Open(testmodels.Path(t, testmodels.SmartTurn), gophonic.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if !gophonic.Supports[speech.TurnDetector](model) || gophonic.Supports[speech.Transcriber](model) {
		t.Fatalf("smart-turn provides %v", model.Provides())
	}
	classifier, err := gophonic.Lane[speech.AudioClassifier](model)
	if err != nil {
		t.Fatal(err)
	}
	defer classifier.Close()
	probs := make([]float32, 2)
	pcm := readFloatFixture(t, "testdata/whisper_jfk.pcm.f32le")
	if err := classifier.ClassifyInto(pcm, 16000, 1, probs); err != nil {
		t.Fatal(err)
	}
	if got := probs[0] + probs[1]; math.Abs(float64(got-1)) > 1e-6 || classifier.Labels()[1] != "complete" {
		t.Fatalf("labels %v, probabilities %v", classifier.Labels(), probs)
	}
	if n := testing.AllocsPerRun(3, func() { _ = classifier.ClassifyInto(pcm, 16000, 1, probs) }); n != 0 {
		t.Fatalf("ClassifyInto allocated %.1f times", n)
	}
}

// A Qwen3 language model opens as a zero-shot text classifier.
func TestOpenQwen3ZeroShot(t *testing.T) {
	model, err := gophonic.Open(testmodels.Path(t, testmodels.Qwen3), gophonic.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	zeroShot, err := gophonic.Lane[speech.ZeroShot](model)
	if err != nil {
		t.Fatal(err)
	}
	labels := []string{"calm or neutral", "happy or excited", "frustrated or angry"}
	classifier, err := zeroShot.Classifier("What is the speaker's mood?", labels)
	if err != nil {
		t.Fatal(err)
	}
	probs := make([]float32, len(labels))
	for text, want := range map[string]int{
		"Can you send me the invoice by Friday?":                 0,
		"Wow, that's great news!":                                1,
		"I've been waiting two hours and nobody called me back.": 2,
	} {
		if err := classifier.ClassifyInto(context.Background(), text, probs); err != nil {
			t.Fatal(err)
		}
		if best := slices.Index(probs, slices.Max(probs)); best != want {
			t.Errorf("%q: %v, want %q", text, probs, labels[want])
		}
	}
}

// A weight format Open does not know fails before any model loads.
func TestOpenChecksFormat(t *testing.T) {
	if _, err := gophonic.Open("anything", gophonic.Options{Format: "fp8"}); !errors.Is(err, speech.ErrUnsupported) {
		t.Fatalf("Open with an unknown format: %v, want speech.ErrUnsupported", err)
	}
}

// A format's model must provide what the format declares: Open fails, and
// closes the model, when it does not.
func TestOpenChecksProvides(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.liar")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	closed := false
	gophonic.Register(gophonic.Format{
		Name:  "liar",
		Match: func(p string) bool { return strings.HasSuffix(p, ".liar") },
		Open: func(p string, _ gophonic.Options) (*gophonic.Model, error) {
			m := gophonic.NewModel("liar", p, func() error { closed = true; return nil })
			return gophonic.Provide(m, func() (moderator, error) { return fixedDetector{}, nil }), nil
		},
		Provides: []reflect.Type{reflect.TypeFor[moderator](), reflect.TypeFor[speech.Transcriber]()},
	})
	if _, err := gophonic.Open(path, gophonic.Options{}); err == nil || !closed {
		t.Fatalf("Open of a model that lacks a declared lane: %v, closed %v", err, closed)
	}
}
