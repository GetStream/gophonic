// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech

// Duplex is a conversational agent that listens and speaks at the same
// time: there are no turns in its interface. Audio flows in and out one
// frame at a time on the caller's clock, and the agent decides many times a
// second whether to keep listening, answer, acknowledge, or stop because it
// was interrupted. An implementation may compose a transcriber, a language
// model, and a synthesizer (a cascade), or run one native speech-to-speech
// model; callers cannot tell.
//
// Step does no inference itself: it hands the input to the agent's
// workers, returns the next frame of its speech, and never blocks on a
// model, so it can run on a real-time audio thread. A Duplex is not safe
// for concurrent use, except that Say and Note may be called from any
// goroutine.
type Duplex interface {
	// Rates reports the sample rates of the mono input and output.
	Rates() (in, out int)
	// Frame reports the samples of one step's input and output, the same
	// span of time.
	Frame() (in, out int)
	// Step consumes one frame of the audio the agent hears and writes one
	// frame of the audio it speaks to out, silence when it is not
	// speaking. It reports the agent's state after the frame. A nil in
	// means no audio arrived for the frame, as when a packet is late: the
	// agent keeps speaking but does not take the gap for silence. Warm
	// steps allocate nothing.
	Step(in, out []float32) (DuplexState, error)
	// Say makes the agent speak text as soon as it can, as when an
	// announcement arrives outside the conversation.
	Say(text string) error
	// Note tells the agent, in words, that something happened outside the
	// conversation: a chat message, someone joining or leaving, a timer.
	// It joins the agent's context, and the agent may answer it.
	Note(text string) error
	// Close stops the agent's workers and releases its lanes.
	Close() error
}

// DuplexState is what a Duplex is doing.
type DuplexState uint8

const (
	// Listening: the agent is silent and attending.
	Listening DuplexState = iota
	// Thinking: the agent decided to answer and is preparing its speech.
	Thinking
	// Speaking: the agent's speech is in out.
	Speaking
)

func (s DuplexState) String() string {
	switch s {
	case Listening:
		return "listening"
	case Thinking:
		return "thinking"
	case Speaking:
		return "speaking"
	}
	return "unknown"
}
