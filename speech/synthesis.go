// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech

import (
	"context"
	"errors"
	"io"
	"unsafe"
)

// Synthesizer turns text into speech as the text arrives, so speech can
// start while a language model is still writing it. It is a lane that
// speaks one utterance at a time: Begin starts one, Write adds its text,
// End says the text is complete, and Read pulls its audio as it is
// decoded. Text is pushed and audio pulled, on any two goroutines, and
// neither waits for the other: Write buffers the text, and Read returns
// what is decoded. Warm utterances allocate nothing.
type Synthesizer interface {
	// SampleRate reports the rate of the PCM Read produces.
	SampleRate() int
	// Voices lists the voices SpeakOptions.Voice may name.
	Voices() []string
	// Begin starts an utterance in opts's voice, dropping the one in
	// progress, as an interruption does. The utterance is cut when ctx is
	// done. It fails, before any speech, on a voice or language the
	// synthesizer lacks.
	Begin(ctx context.Context, opts SpeakOptions) error
	// Write adds text to the utterance. It copies the text and returns
	// without waiting for it to be spoken.
	Write(text []byte) (int, error)
	// End marks the utterance's text complete.
	End() error
	// Read fills pcm with the utterance's next samples and reports how
	// many, waiting until some are decoded. After the last sample it
	// returns io.EOF, and ctx's error once the utterance is cut.
	Read(pcm []float32) (int, error)
	// Voiced reports how many bytes of the utterance's text the samples
	// read so far have spoken, as the synthesizer follows the text: it
	// never decreases, and after io.EOF it is all of the text. Captions
	// show the text up to it, and an interrupted speaker keeps what was
	// heard. It may be called from any goroutine.
	Voiced() int
	// Close releases the lane. It is safe to call more than once.
	Close() error
}

// SpeakOptions selects how text is spoken.
type SpeakOptions struct {
	// Voice names one of Synthesizer.Voices; empty picks the model's
	// default.
	Voice string
	// Language is the language to speak, as an ISO 639-1 code or an English
	// name; empty lets the model follow the text.
	Language string
	// Style says how to speak, in words: "Calm and even; never laugh."
	// Synthesizers that take no instruction ignore it.
	Style string
}

// Synthesize speaks text in one utterance and appends its samples to dst:
// the whole-text path of tools and tests.
func Synthesize(ctx context.Context, s Synthesizer, opts SpeakOptions, text string, dst []float32) ([]float32, error) {
	if err := s.Begin(ctx, opts); err != nil {
		return dst, err
	}
	// Write copies the text, so it may read the string's bytes.
	if _, err := s.Write(unsafe.Slice(unsafe.StringData(text), len(text))); err != nil {
		return dst, err
	}
	if err := s.End(); err != nil {
		return dst, err
	}
	for {
		if len(dst) == cap(dst) {
			dst = append(dst, 0)[:len(dst)]
		}
		n, err := s.Read(dst[len(dst):cap(dst)])
		dst = dst[:len(dst)+n]
		if errors.Is(err, io.EOF) {
			return dst, nil
		}
		if err != nil {
			return dst, err
		}
	}
}
