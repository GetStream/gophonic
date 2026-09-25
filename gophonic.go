// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package gophonic runs speech models in pure Go: speech recognition
// (Qwen3-ASR, Whisper) and end-of-turn detection (Smart Turn, TinyMelNet).
//
// Open loads any registered model by path and reports what it does; its
// lanes implement the model-independent interfaces of package speech. The
// built-in formats are registered already, and Register adds more. Each
// model also has its own package (qwen3asr, whisper, smartturn, tinymel)
// with lower level entry points.
package gophonic

import (
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/GetStream/gophonic/speech"
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
	close           func() error
}

// NewTranscriptionModel returns a transcription Model named name whose
// lanes come from newLane. close, when not nil, releases the model's
// resources (such as GPU memory) and runs once, from Model.Close. A Format's
// Open builds its Model with this or NewTurnDetectionModel.
func NewTranscriptionModel(name string, newLane func() (speech.Transcriber, error), close func() error) *Model {
	return &Model{name: name, kind: Transcription, newTranscriber: newLane, close: close}
}

// NewTurnDetectionModel is NewTranscriptionModel for turn detectors.
func NewTurnDetectionModel(name string, newLane func() (speech.TurnDetector, error), close func() error) *Model {
	return &Model{name: name, kind: TurnDetection, newTurnDetector: newLane, close: close}
}

// Name identifies the architecture, such as "qwen3-asr" or "smart-turn".
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

// Close releases the model's resources. Close its lanes first; neither the
// model nor its lanes may be used afterwards. It is safe to call more than
// once.
func (m *Model) Close() error {
	c := m.close
	m.close, m.newTranscriber, m.newTurnDetector = nil, nil, nil
	if c == nil {
		return nil
	}
	return c()
}

// A Format is one kind of model that Open recognizes.
type Format struct {
	// Name identifies the format in errors and listings; the Model's Name
	// is the one its Open gives it.
	Name string
	// Match reports whether path holds this format. It should read as
	// little as it can: a file signature, or one field of a config file.
	Match func(path string) bool
	// Open loads the model at path.
	Open func(path string, opts Options) (*Model, error)
}

// ErrUnknownFormat is returned by Open for a path no format matches.
var ErrUnknownFormat = errors.New("gophonic: unrecognized model format")

var (
	registry sync.RWMutex
	formats  = builtinFormats() // tried in order
)

// Register adds a format for Open. Formats registered later are tried
// first, so a registered format can take over paths a built-in one matches.
// It is safe to call concurrently with Open.
func Register(f Format) {
	if f.Match == nil || f.Open == nil {
		panic("gophonic: Register of a format without Match or Open")
	}
	registry.Lock()
	formats = append([]Format{f}, formats...)
	registry.Unlock()
}

// Formats lists the registered formats, in the order Open tries them.
func Formats() []Format {
	registry.RLock()
	defer registry.RUnlock()
	return slices.Clone(formats)
}

// Open loads the model at path with the first registered format that
// matches it: by default, an official Qwen3-ASR checkpoint directory or a
// converted Whisper, Smart Turn, or TinyMelNet .gophonic bundle.
func Open(path string, opts Options) (*Model, error) {
	if opts.Threads < 0 {
		return nil, fmt.Errorf("gophonic: invalid thread count %d", opts.Threads)
	}
	for _, f := range Formats() {
		if f.Match(path) {
			return f.Open(path, opts)
		}
	}
	if err := exists(path); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("%w: %s", ErrUnknownFormat, path)
}
