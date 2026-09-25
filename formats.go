// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"io"
	"os"

	"github.com/GetStream/gophonic/qwen3asr"
	"github.com/GetStream/gophonic/smartturn"
	"github.com/GetStream/gophonic/speech"
	"github.com/GetStream/gophonic/tinymel"
	"github.com/GetStream/gophonic/whisper"
)

// builtinFormats are the models gophonic ships, in the order Open tries
// them.
func builtinFormats() []Format {
	return []Format{
		{Name: "qwen3-asr", Match: qwen3asr.IsModelDir, Open: openQwen3ASR},
		{Name: "whisper", Match: signature(whisper.BundleMagic), Open: openWhisper},
		{Name: "smart-turn", Match: signature(smartturn.BundleMagic), Open: openSmartTurn},
		{Name: "tinymel", Match: signature(tinymel.BundleMagic), Open: openTinyMel},
	}
}

func openQwen3ASR(path string, opts Options) (*Model, error) {
	m, err := qwen3asr.Load(path, qwen3asr.Options{})
	if err != nil {
		return nil, err
	}
	return NewTranscriptionModel("qwen3-asr", func() (speech.Transcriber, error) {
		return qwen3asr.NewTranscriber(m, opts.Threads)
	}, func() error { m.Release(); return nil }), nil
}

func openWhisper(path string, opts Options) (*Model, error) {
	m, err := whisper.Load(path)
	if err != nil {
		return nil, err
	}
	return NewTranscriptionModel("whisper", func() (speech.Transcriber, error) {
		if opts.Threads == 0 {
			return whisper.NewTranscriber(m)
		}
		return whisper.NewTranscriberWithWorkers(m, opts.Threads)
	}, nil), nil
}

func openSmartTurn(path string, _ Options) (*Model, error) {
	m, err := smartturn.Load(path)
	if err != nil {
		return nil, err
	}
	return NewTurnDetectionModel("smart-turn", func() (speech.TurnDetector, error) {
		return smartturn.NewSession(m)
	}, nil), nil
}

func openTinyMel(path string, opts Options) (*Model, error) {
	m, err := tinymel.Load(path)
	if err != nil {
		return nil, err
	}
	helpers := max(opts.Threads-1, 0)
	return NewTurnDetectionModel("tinymel", func() (speech.TurnDetector, error) {
		return tinymel.NewSession(m, helpers)
	}, nil), nil
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
