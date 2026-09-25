// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package gophonic runs speech models in pure Go: speech recognition
// (Whisper, Qwen3-ASR) and end-of-turn detection (Smart Turn, TinyMelNet).
//
// Open loads any supported model by path and reports what it does; its
// lanes implement the model-independent interfaces of package speech. Each
// model also has its own package (whisper, smartturn, tinymel) with lower
// level entry points.
package gophonic

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/GetStream/gophonic/smartturn"
	"github.com/GetStream/gophonic/speech"
	"github.com/GetStream/gophonic/tinymel"
	"github.com/GetStream/gophonic/whisper"
)

// Kind is what a Model does.
type Kind uint8

const (
	// Transcription models turn speech into text.
	Transcription Kind = iota + 1
	// TurnDetection models predict whether a speaker has finished.
	TurnDetection
)

func (k Kind) String() string {
	switch k {
	case Transcription:
		return "transcription"
	case TurnDetection:
		return "turn detection"
	}
	return fmt.Sprintf("Kind(%d)", uint8(k))
}

// Options configures Open. The zero value picks each model's defaults.
type Options struct {
	// Threads bounds the CPU workers of each lane, including the caller.
	// Zero picks the model's default.
	Threads int
}

// Model is a loaded speech model. Its weights are immutable and shared by
// every lane opened from it; lanes own their scratch and may run
// concurrently with each other.
type Model struct {
	name            string
	kind            Kind
	newTranscriber  func() (speech.Transcriber, error)
	newTurnDetector func() (speech.TurnDetector, error)
}

// Name identifies the architecture, such as "whisper" or "smart-turn".
func (m *Model) Name() string { return m.name }

// Kind reports what the model does.
func (m *Model) Kind() Kind { return m.kind }

// NewTranscriber opens a transcription lane. It fails with
// speech.ErrUnsupported for a turn-detection model.
func (m *Model) NewTranscriber() (speech.Transcriber, error) {
	if m.newTranscriber == nil {
		return nil, fmt.Errorf("gophonic: %s does not transcribe: %w", m.name, speech.ErrUnsupported)
	}
	return m.newTranscriber()
}

// NewTurnDetector opens a turn-detection lane. It fails with
// speech.ErrUnsupported for a transcription model.
func (m *Model) NewTurnDetector() (speech.TurnDetector, error) {
	if m.newTurnDetector == nil {
		return nil, fmt.Errorf("gophonic: %s does not detect turns: %w", m.name, speech.ErrUnsupported)
	}
	return m.newTurnDetector()
}

// ErrUnknownFormat is returned by Open for a file it does not recognize.
var ErrUnknownFormat = errors.New("gophonic: unrecognized model format")

// Open loads the model at path, recognizing its format from the file: a
// converted Whisper, Smart Turn, or TinyMelNet .gophonic bundle.
func Open(path string, opts Options) (*Model, error) {
	if opts.Threads < 0 {
		return nil, fmt.Errorf("gophonic: invalid thread count %d", opts.Threads)
	}
	magic, err := readMagic(path)
	if err != nil {
		return nil, err
	}
	switch magic {
	case whisper.BundleMagic:
		model, err := whisper.Load(path)
		if err != nil {
			return nil, err
		}
		workers := opts.Threads
		return &Model{name: "whisper", kind: Transcription, newTranscriber: func() (speech.Transcriber, error) {
			if workers == 0 {
				return whisper.NewTranscriber(model)
			}
			return whisper.NewTranscriberWithWorkers(model, workers)
		}}, nil
	case smartturn.BundleMagic:
		model, err := smartturn.Load(path)
		if err != nil {
			return nil, err
		}
		return &Model{name: "smart-turn", kind: TurnDetection, newTurnDetector: func() (speech.TurnDetector, error) {
			return smartturn.NewSession(model)
		}}, nil
	case tinymel.BundleMagic:
		model, err := tinymel.Load(path)
		if err != nil {
			return nil, err
		}
		helpers := max(opts.Threads-1, 0)
		return &Model{name: "tinymel", kind: TurnDetection, newTurnDetector: func() (speech.TurnDetector, error) {
			return tinymel.NewSession(model, helpers)
		}}, nil
	}
	return nil, fmt.Errorf("%w: %s", ErrUnknownFormat, path)
}

func readMagic(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var magic [8]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return "", fmt.Errorf("%w: %s: %v", ErrUnknownFormat, path, err)
	}
	return string(magic[:]), nil
}
