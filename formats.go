// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"errors"
	"io"
	"os"
	"reflect"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/qwen3"
	"github.com/GetStream/gophonic/qwen3asr"
	"github.com/GetStream/gophonic/qwen3tts"
	"github.com/GetStream/gophonic/smartturn"
	"github.com/GetStream/gophonic/speech"
	"github.com/GetStream/gophonic/tinymel"
	"github.com/GetStream/gophonic/whisper"
)

// builtinFormats are the models gophonic ships, in the order Open tries
// them.
func builtinFormats() []Format {
	transcriber := []reflect.Type{reflect.TypeFor[speech.Transcriber]()}
	turns := []reflect.Type{reflect.TypeFor[speech.TurnDetector](), reflect.TypeFor[speech.AudioClassifier]()}
	return []Format{
		{Name: "qwen3-asr", Match: qwen3asr.IsModelDir, Open: openQwen3ASR, Provides: transcriber},
		{Name: "qwen3-tts", Match: qwen3tts.IsModelDir, Open: openQwen3TTS, Provides: []reflect.Type{reflect.TypeFor[speech.Synthesizer]()}},
		{Name: "qwen3", Match: qwen3.IsModelDir, Open: openQwen3, Provides: []reflect.Type{reflect.TypeFor[chat.Generator](), reflect.TypeFor[speech.ZeroShot]()}},
		{Name: "whisper", Match: signature(whisper.BundleMagic), Open: openWhisper, Provides: transcriber},
		{Name: "smart-turn", Match: signature(smartturn.BundleMagic), Open: openSmartTurn, Provides: turns},
		{Name: "tinymel", Match: signature(tinymel.BundleMagic), Open: openTinyMel, Provides: turns},
	}
}

func openQwen3ASR(path string, opts Options) (*Model, error) {
	m, err := qwen3asr.Load(path, qwen3asr.Options{})
	if err != nil {
		return nil, err
	}
	release := func() error { m.Release(); return nil }
	return Provide(NewModel("qwen3-asr", release), func() (speech.Transcriber, error) {
		return qwen3asr.NewTranscriber(m, opts.Threads)
	}), nil
}

func openQwen3TTS(path string, opts Options) (*Model, error) {
	m, err := qwen3tts.Load(path, qwen3tts.Options{Threads: opts.Threads})
	if err != nil {
		return nil, err
	}
	release := func() error { m.Release(); return nil }
	return Provide(NewModel("qwen3-tts", release), func() (speech.Synthesizer, error) {
		return qwen3tts.NewSynthesizer(m)
	}), nil
}

// A Qwen3 language model generates text (chat.Generator) and answers
// questions about text (speech.ZeroShot, whose classifiers are prepared
// multiple-choice questions), both from one loaded copy of its weights.
func openQwen3(path string, opts Options) (*Model, error) {
	g, err := qwen3.OpenChat(path, qwen3.Options{Threads: opts.Threads})
	if err != nil {
		return nil, err
	}
	m, err := g.Questions(qwen3.Options{Threads: opts.Threads})
	if err != nil {
		g.Close()
		return nil, err
	}
	model := NewModel("qwen3", func() error { return errors.Join(m.Close(), g.Close()) })
	Provide(model, func() (chat.Generator, error) { return generator{g}, nil })
	return Provide(model, func() (speech.ZeroShot, error) { return zeroShot{m}, nil }), nil
}

// zeroShot is one lane of a shared Qwen3 model, which serializes its calls;
// closing the lane leaves the model open.
type zeroShot struct{ m *qwen3.Model }

func (z zeroShot) Classifier(question string, labels []string) (speech.TextClassifier, error) {
	return z.m.Classifier(question, labels)
}

func (zeroShot) Close() error { return nil }

// generator is one lane of a shared Qwen3 generator; closing the lane
// leaves it open.
type generator struct{ g *qwen3.Chat }

func (g generator) NewSession(system string) (chat.Session, error) { return g.g.NewSession(system) }

func (generator) Close() error { return nil }

func openWhisper(path string, opts Options) (*Model, error) {
	m, err := whisper.Load(path)
	if err != nil {
		return nil, err
	}
	return Provide(NewModel("whisper", m.Close), func() (speech.Transcriber, error) {
		if opts.Threads == 0 {
			return whisper.NewTranscriber(m)
		}
		return whisper.NewTranscriberWithWorkers(m, opts.Threads)
	}), nil
}

// The turn detectors are also binary audio classifiers.
func openSmartTurn(path string, _ Options) (*Model, error) {
	m, err := smartturn.Load(path)
	if err != nil {
		return nil, err
	}
	model := NewModel("smart-turn", nil)
	Provide(model, func() (speech.TurnDetector, error) { return smartturn.NewSession(m) })
	return Provide(model, func() (speech.AudioClassifier, error) { return smartturn.NewSession(m) }), nil
}

func openTinyMel(path string, opts Options) (*Model, error) {
	m, err := tinymel.Load(path)
	if err != nil {
		return nil, err
	}
	helpers := max(opts.Threads-1, 0)
	model := NewModel("tinymel", nil)
	Provide(model, func() (speech.TurnDetector, error) { return tinymel.NewSession(m, helpers) })
	return Provide(model, func() (speech.AudioClassifier, error) { return tinymel.NewSession(m, helpers) }), nil
}

// signature matches files that start with the eight bytes magic.
func signature(magic string) func(path string) bool {
	return func(path string) bool {
		f, err := os.Open(path)
		if err != nil {
			return false
		}
		defer f.Close()
		var head [8]byte
		_, err = io.ReadFull(f, head[:])
		return err == nil && string(head[:]) == magic
	}
}

// exists reports why path cannot be opened, if it cannot.
func exists(path string) error {
	_, err := os.Stat(path)
	return err
}
