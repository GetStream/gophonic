// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
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
	if model.Name() != "whisper" || model.Kind() != gophonic.Transcription {
		t.Fatalf("Open = %s %v, want whisper transcription", model.Name(), model.Kind())
	}
	if _, err := model.NewTurnDetector(); !errors.Is(err, speech.ErrUnsupported) {
		t.Fatalf("NewTurnDetector error = %v, want speech.ErrUnsupported", err)
	}
	lane, err := model.NewTranscriber()
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
			return gophonic.NewTurnDetectionModel("custom", func() (speech.TurnDetector, error) {
				return fixedDetector{}, nil
			}, func() error { closed++; return nil }), nil
		},
	})
	if f := gophonic.Formats(); f[0].Name != "custom" || f[len(f)-1].Name != "tinymel" {
		t.Fatalf("formats %v", f)
	}
	model, err := gophonic.Open(path, gophonic.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "custom" || model.Kind() != gophonic.TurnDetection {
		t.Fatalf("Open = %s %v", model.Name(), model.Kind())
	}
	detector, err := model.NewTurnDetector()
	if err != nil {
		t.Fatal(err)
	}
	if p, err := detector.PredictInto(nil, 16000, 1); err != nil || !p.Complete {
		t.Fatalf("PredictInto = %+v, %v", p, err)
	}
	if _, err := model.NewTranscriber(); !errors.Is(err, speech.ErrUnsupported) {
		t.Fatalf("NewTranscriber error = %v, want speech.ErrUnsupported", err)
	}
	if model.Close() != nil || model.Close() != nil || closed != 1 {
		t.Fatalf("Close ran %d times", closed)
	}
}

type fixedDetector struct{}

func (fixedDetector) PredictInto([]float32, int, int) (speech.Prediction, error) {
	return speech.Prediction{Probability: 1, Complete: true}, nil
}

func (fixedDetector) Close() error { return nil }
