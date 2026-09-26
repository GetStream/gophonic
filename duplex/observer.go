// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package duplex

import "time"

// Observer receives what the cascade hears and says, each reply's stages,
// and its errors, from worker goroutines. Embed Base to implement only
// some of it. Text slices belong to the cascade and are valid during the
// call.
type Observer interface {
	// Heard reports a speaker's words: partial while they are spoken, then
	// final once the utterance is answered (in words or by silence).
	Heard(speaker string, text []byte, final bool)
	// Said reports the agent's reply as it grows: the text so far, how
	// many of its bytes the voice has spoken, and whether the reply is
	// finished or cut. Captions show text[:voiced]; a record keeps the
	// final text.
	Said(text []byte, voiced int, final bool)
	// Stage reports a reply's progress since the speaker's pause.
	Stage(s Stage, elapsed time.Duration)
	// Error reports an error of a worker.
	Error(err error)
}

// Base is an Observer that observes nothing; embed it and override.
type Base struct{}

func (Base) Heard(string, []byte, bool) {}
func (Base) Said([]byte, int, bool)     {}
func (Base) Stage(Stage, time.Duration) {}
func (Base) Error(error)                {}

// Stage is a step of a reply.
type Stage uint8

const (
	// Transcribed: the utterance is transcribed.
	Transcribed Stage = iota + 1
	// Judged: the language model judged whether the words are finished.
	Judged
	// FirstText: the model wrote its first words.
	FirstText
	// FirstAudio: the voice produced its first frame.
	FirstAudio
	// TurnOver: the speaker's turn was found over; the answer plays from
	// the later of this and FirstAudio.
	TurnOver
	// Silent: the model chose to say nothing.
	Silent
	// Planned: the model asked to be asked again after a time.
	Planned
	// Called: a tool ran.
	Called
	// Dropped: a reply was superseded, or its repetition of the message
	// was cut.
	Dropped
	// Interrupted: speech over the agent stopped it.
	Interrupted
	// Continued: speech over the agent, an acknowledgement, let it go on.
	Continued
	// Overruled: a silence chosen when no one asked for quiet was
	// overruled, and the model answers after all.
	Overruled
)

var stageNames = [...]string{"", "transcribed", "judged", "first text", "first audio", "turn over",
	"silent", "planned", "called", "dropped", "interrupted", "continued", "overruled"}

func (s Stage) String() string {
	if int(s) < len(stageNames) {
		return stageNames[s]
	}
	return "unknown"
}
