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
