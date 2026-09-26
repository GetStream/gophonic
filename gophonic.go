// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package gophonic runs speech and language models in pure Go: speech
// recognition (Qwen3-ASR, Whisper), speech synthesis (Qwen3-TTS),
// end-of-turn detection (Smart Turn, TinyMelNet), and Qwen3 language
// models.
//
// Open loads any registered model by path; the model provides lanes of the
// interface types it supports, such as those of packages speech and chat.
// The built-in formats are registered already, and Register adds more. Each
// model also has its own package (qwen3, qwen3asr, qwen3tts, whisper,
// smartturn, tinymel) with lower level entry points.
package gophonic

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"

	"github.com/GetStream/gophonic/internal/wcache"
	"github.com/GetStream/gophonic/speech"
)

// Options configures Open. The zero value picks each model's defaults.
type Options struct {
	// Threads bounds the CPU workers of each lane, including the caller.
	// Zero picks the model's default.
	Threads int
	// Format names the weight format of the models that offer a choice, the
	// Qwen models: one of the Format constants. Empty picks the fastest
	// format of llama.cpp Q8_0 fidelity on this machine, the Apple GPU
	// where Metal is present and exact weights on the CPU elsewhere. Models
	// with one format ignore it; a Qwen model without the format named fails
	// with speech.ErrUnsupported.
	Format string
}

// Weight formats for Options.Format. Every format but FormatF16 is checked
// against the checkpoint's BF16 hidden states, and all but FormatGPUQ4 are
// at llama.cpp Q8_0 fidelity or better.
const (
	FormatF16   = "f16"    // every BF16 weight exactly, on the CPU's matrix units
	FormatInt8  = "int8"   // int8 rows in a rotated basis, on the CPU
	FormatGPU   = "gpu"    // int8 rows in a rotated basis, on the Apple GPU
	FormatGPUQ8 = "gpu-q8" // int8 blocks of 32, on the Apple GPU
	FormatGPUQ4 = "gpu-q4" // 4-bit blocks of 32, on the Apple GPU: lower fidelity
)

// CacheDir is where prepared weights are cached, so that a checkpoint is
// quantized once and mapped in place afterwards: $GOPHONIC_CACHE, or
// gophonic in the user cache directory. Empty means no cache: an empty
// $GOPHONIC_CACHE, or no cache directory, prepares weights at every load.
func CacheDir() string { return wcache.Dir }

// Model is a loaded model. Its weights are immutable and shared by every
// lane opened from it; lanes own their scratch and may run concurrently with
// each other.
//
// What a model can do is the set of lane types it provides: any interface,
// such as speech.Transcriber, speech.TurnDetector, speech.AudioClassifier,
// speech.TextClassifier, speech.ZeroShot, or one a third-party package
// defines. Lane opens one of them, and Supports reports whether it can.
type Model struct {
	name, path string
	lanes      []lane
	close      func() error
}

// lane opens lanes of one provided type.
type lane struct {
	typ  reflect.Type
	open func() (any, error)
}

// NewModel returns a model of architecture name, loaded from path, with no
// lanes; Provide adds them. close, when not nil, releases the model's
// resources (such as GPU memory) and runs once, from Model.Close. A
// Format's Open builds its Model this way.
func NewModel(name, path string, close func() error) *Model {
	return &Model{name: name, path: path, close: close}
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

// Path is the path the model was loaded from.
func (m *Model) Path() string { return m.path }

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
	// they are known only once a model is open. Open checks that a model
	// provides them.
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
	switch opts.Format {
	case "", FormatF16, FormatInt8, FormatGPU, FormatGPUQ8, FormatGPUQ4:
	default:
		return nil, fmt.Errorf("gophonic: unknown weight format %q: %w", opts.Format, speech.ErrUnsupported)
	}
	if f, ok := Detect(path); ok {
		m, err := f.Open(path, opts)
		if err != nil {
			return nil, err
		}
		for _, t := range f.Provides {
			if !slices.ContainsFunc(m.lanes, func(l lane) bool { return l.typ == t }) {
				m.Close()
				return nil, fmt.Errorf("gophonic: format %s declares %v, but its model does not provide it", f.Name, t)
			}
		}
		return m, nil
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
