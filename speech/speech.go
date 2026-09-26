// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package speech defines what gophonic's models have in common: transcripts,
// end-of-turn predictions, class probabilities, and the Transcriber,
// TurnDetector, AudioClassifier, TextClassifier, and ZeroShot interfaces
// that applications, the CLI, and the HTTP server program against. A model
// may provide any of them, or interfaces of its own (see gophonic.Lane).
//
// Model packages implement these interfaces (whisper and qwen3asr
// transcribe; smartturn and tinymel detect turns and classify audio; qwen3
// classifies text by question), and gophonic.Open loads any registered model
// by path. This package holds no model code, so any package, including a
// third-party backend, can implement the interfaces.
package speech

import (
	"context"
	"errors"
)

// SampleRate is the rate of the PCM every Transcriber accepts: mono float32
// samples at 16 kHz.
const SampleRate = 16000

var (
	// ErrClosed is returned by calls on a closed Transcriber or TurnDetector.
	ErrClosed = errors.New("speech: closed")
	// ErrUnsupported is wrapped by errors for options a model cannot honor,
	// such as a language it does not know or timing it cannot produce.
	ErrUnsupported = errors.New("speech: unsupported option")
)

// Transcriber turns speech into text. A Transcriber owns the scratch of one
// call lane: calls on it must not overlap, so open one per concurrent lane.
// Model weights can be shared by several Transcribers.
type Transcriber interface {
	// Transcribe transcribes mono 16 kHz PCM into dst, reusing the capacity
	// of dst's slices. Warm calls whose results fit that capacity do not
	// allocate.
	Transcribe(ctx context.Context, pcm []float32, opts Options, dst *Transcript) error
	// Close releases the lane's workers and scratch. It is safe to call
	// more than once.
	Close() error
}

// Options adjusts one transcription. The zero value detects the language
// and returns text only.
type Options struct {
	// Language is the spoken language as an ISO 639-1 code ("en") or an
	// English name ("English"). Empty lets the model detect it.
	Language string
	// Languages, when Language is empty, are the languages that may be
	// spoken: the model detects the likeliest of them, as when a call is
	// in English and Portuguese and a noise must not come out as Chinese.
	// Transcribers that detect no language ignore it.
	Languages []string
	// Context is text that primes recognition, such as names or terms that
	// occur in the audio.
	// Transcribers do not retain Options after Transcribe returns.
	Context string
	// Segments requests timed segments in Transcript.Segments.
	Segments bool
	// Words requests word timing in Transcript.Words. It implies Segments.
	Words bool
	// Partial, when not nil, is an earlier transcript of the beginning of
	// the same audio, as when speech is transcribed while it is spoken. A
	// transcriber that can checks it against the audio in one pass and
	// decodes only where it differs or ends, instead of starting over; the
	// transcript is the same either way. Others ignore it.
	Partial *Transcript
	// Turn requests Transcript.Turn: whether the speaker's turn ends where
	// the audio does, judged from what was said and how. Transcribers that
	// cannot judge it return an error wrapping ErrUnsupported.
	Turn bool
}

// Transcript is the result of one transcription. Offsets index Text.
type Transcript struct {
	Text []byte
	// Language is the English name of the detected or requested language,
	// or empty if the model does not report one.
	Language string
	Segments []Segment
	Words    []Word
	// Turn, when requested, is whether the speaker's turn ends where the
	// audio does. Audio without speech has none to end.
	Turn Prediction
}

// Reset empties t while keeping its capacity.
func (t *Transcript) Reset() {
	t.Text = t.Text[:0]
	t.Language = ""
	t.Segments = t.Segments[:0]
	t.Words = t.Words[:0]
	t.Turn = Prediction{}
}

// Segment is a timed span of a Transcript.
type Segment struct {
	Start, End         float64 // seconds from the start of the audio
	TextStart, TextEnd int     // byte range in Transcript.Text
	WordStart, WordEnd int     // range in Transcript.Words, when words were requested
}

// Word is one aligned word of a Transcript.
type Word struct {
	Start, End         float64 // seconds from the start of the audio
	TextStart, TextEnd int     // byte range in Transcript.Text
	Probability        float64 // the model's confidence, or 0 if unknown
}

// Prediction is an end-of-turn decision.
type Prediction struct {
	// Probability is the model's probability that the speaker has finished.
	Probability float32 `json:"probability"`
	// Complete reports whether Probability reaches the model's threshold.
	Complete bool `json:"complete"`
}

// AudioClassifier assigns a probability to each of its labels for a
// stretch of audio: whether a turn is complete, an emotion, a language, a
// sound event. PCM is interleaved mono or stereo at 8–96 kHz. An
// AudioClassifier owns the scratch of one call lane: calls must not overlap,
// so open one per concurrent lane.
type AudioClassifier interface {
	// Labels names the classes in the order ClassifyInto writes them. The
	// slice belongs to the classifier and does not change.
	Labels() []string
	// ClassifyInto writes one probability per label to probs, which has
	// len(Labels()) values. Warm calls do not allocate.
	ClassifyInto(pcm []float32, sampleRate, channels int, probs []float32) error
	// Close releases the lane's scratch. It is safe to call more than once.
	Close() error
}

// TextClassifier is AudioClassifier for text: sentiment, intent, topic, or
// the answer to a question a language model was asked.
type TextClassifier interface {
	Labels() []string
	ClassifyInto(ctx context.Context, text string, probs []float32) error
	Close() error
}

// ZeroShot builds text classifiers from a question and its candidate
// answers, as a language model answers a multiple-choice question: the
// labels are the answers, and a classifier prepared once scores any number
// of texts. Moderation, intent, or sentiment with labels chosen at run time
// all fit it.
type ZeroShot interface {
	// Classifier prepares a classifier for question whose labels are
	// labels. It may allocate; reuse the result.
	Classifier(question string, labels []string) (TextClassifier, error)
	// Close releases the lane. It is safe to call more than once.
	Close() error
}

// TurnDetector predicts from the latest audio whether a speaker has finished
// their turn. PCM is interleaved mono or stereo at 8–96 kHz.
//
// Implementations own the mutable scratch of one call lane: Predict and
// Close must not overlap on one TurnDetector, so open one per concurrent
// lane. A custom backend can implement this interface without registering
// itself anywhere.
type TurnDetector interface {
	// Predict judges the latest audio. Warm calls allocate nothing.
	Predict(pcm []float32, sampleRate, channels int) (Prediction, error)
	Close() error
}
