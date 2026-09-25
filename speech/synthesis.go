// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech

import "context"

// Synthesizer turns text into speech as the text arrives, so speech can
// start while a language model is still writing it. A Synthesizer owns the
// scratch of one lane: calls on it must not overlap.
type Synthesizer interface {
	// SampleRate reports the rate of the PCM Speak produces.
	SampleRate() int
	// Voices lists the voices SpeakOptions.Voice may name.
	Voices() []string
	// Speak speaks the text that next returns, piece by piece, passing mono
	// PCM to out as soon as it is decoded; out must not retain the slice.
	// next blocks until more text is available and returns io.EOF once the
	// text is complete; Speak calls it while earlier text is being spoken.
	// Speak returns when the speech ends, when ctx is done (as when a
	// listener interrupts), or with the first error of next or out. Warm
	// calls allocate nothing.
	Speak(ctx context.Context, opts SpeakOptions, next func() ([]byte, error), out func(pcm []float32) error) error
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
}
