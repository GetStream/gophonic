// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package gophonic runs speech models in pure Go: speech recognition
// (Qwen3-ASR, Whisper) and end-of-turn detection (Smart Turn, TinyMelNet).
//
// Open loads any registered model by path; the model provides lanes of the
// interface types it supports, such as those of package speech. The
// built-in formats are registered already, and Register adds more. Each
// model also has its own package (qwen3asr, whisper, smartturn, tinymel)
// with lower level entry points.
package gophonic

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"

	"github.com/GetStream/gophonic/speech"
)

// Options configures Open. The zero value picks each model's defaults.
type Options struct {
	// Threads bounds the CPU workers of each lane, including the caller.
	// Zero picks the model's default.
	Threads int
}

// Model is a loaded model. Its weights are immutable and shared by every
// lane opened from it; lanes own their scratch and may run concurrently with
// each other.
//
// What a model can do is the set of lane types it provides: any interface,
// such as speech.Transcriber, speech.TurnDetector, speech.AudioClassifier,
// speech.TextClassifier, speech.ZeroShot, or one a third-party package
// defines. Lane opens one of them, and Supports reports whether it can.
type Model struct {
	name  string
	lanes []lane
	close func() error
}

// lane opens lanes of one provided type.
type lane struct {
	typ  reflect.Type
	open func() (any, error)
}

// NewModel returns a model named name with no lanes; Provide adds them.
// close, when not nil, releases the model's resources (such as GPU memory)
// and runs once, from Model.Close. A Format's Open builds its Model this way.
func NewModel(name string, close func() error) *Model {
	return &Model{name: name, close: close}
}

// Provide makes m open lanes of type T, normally an interface, with open.
// It returns m, so a Format can chain the lanes it provides. Providing a
// type again replaces its constructor.
func Provide[T any](m *Model, open func() (T, error)) *Model {
	typ := reflect.TypeFor[T]()
	fn := func() (any, error) { return open() }
	for i := range m.lanes {
		if m.lanes[i].typ == typ {
			m.lanes[i].open = fn
			return m
		}
	}
	m.lanes = append(m.lanes, lane{typ, fn})
	return m
}

// Lane opens a lane of type T from m, or fails with speech.ErrUnsupported
// when m does not provide that type.
func Lane[T any](m *Model) (T, error) {
	var zero T
	typ := reflect.TypeFor[T]()
	for _, l := range m.lanes {
		if l.typ != typ {
			continue
		}
		v, err := l.open()
		if err != nil {
			return zero, err
		}
		return v.(T), nil
	}
	return zero, fmt.Errorf("gophonic: %s does not provide %v: %w", m.name, typ, speech.ErrUnsupported)
}

// Supports reports whether m provides lanes of type T.
func Supports[T any](m *Model) bool {
	typ := reflect.TypeFor[T]()
	return slices.ContainsFunc(m.lanes, func(l lane) bool { return l.typ == typ })
}

// Name identifies the architecture, such as "qwen3-asr" or "smart-turn".
func (m *Model) Name() string { return m.name }

// Provides lists the lane types m provides, in the order its format
// provided them.
func (m *Model) Provides() []reflect.Type {
	types := make([]reflect.Type, len(m.lanes))
	for i, l := range m.lanes {
		types[i] = l.typ
	}
	return types
}

// Close releases the model's resources. Close its lanes first; neither the
// model nor its lanes may be used afterwards. It is safe to call more than
// once.
func (m *Model) Close() error {
	c := m.close
	m.close, m.lanes = nil, nil
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
	// Provides lists the lane types the format's models provide, so a
	// server can choose a model for a task without loading it; nil means
	// they are known only once a model is open.
	Provides []reflect.Type
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
	if f, ok := Detect(path); ok {
		return f.Open(path, opts)
	}
	if err := exists(path); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("%w: %s", ErrUnknownFormat, path)
}

// Detect returns the format Open would load path with, without loading it.
func Detect(path string) (Format, bool) {
	for _, f := range Formats() {
		if f.Match(path) {
			return f, true
		}
	}
	return Format{}, false
}
