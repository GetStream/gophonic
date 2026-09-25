// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package duplex

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/speech"
)

// Fake lanes: the cascade's behavior, not the models', is under test.
type fakeTranscriber struct{}

func (fakeTranscriber) Transcribe(_ context.Context, _ []float32, _ speech.Options, t *speech.Transcript) error {
	t.Text = append(t.Text[:0], "hello gopher"...)
	return nil
}
func (fakeTranscriber) Close() error { return nil }

type fakeTurns struct{}

func (fakeTurns) PredictInto([]float32, int, int) (speech.Prediction, error) {
	return speech.Prediction{Probability: 1, Complete: true}, nil
}
func (fakeTurns) Close() error { return nil }

type fakeSession struct {
	mu        sync.Mutex
	messages  []string
	truncated int
}

func (s *fakeSession) Add(role chat.Role, text string) error {
	s.mu.Lock()
	s.messages = append(s.messages, text)
	s.mu.Unlock()
	return nil
}

func (s *fakeSession) Reply(ctx context.Context, _ chat.Options, sink func([]byte) error) error {
	for _, p := range []string{"Hi ", "there, ", "friend."} {
		if err := sink([]byte(p)); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.messages = append(s.messages, "Hi there, friend.")
	s.mu.Unlock()
	return ctx.Err()
}

func (s *fakeSession) Truncate(n int) error {
	s.mu.Lock()
	s.truncated = n
	s.mu.Unlock()
	return nil
}
func (s *fakeSession) Prefill(ctx context.Context) error { return ctx.Err() }
func (s *fakeSession) Finished(ctx context.Context, _ chat.Role, _ string) (float32, error) {
	return 1, ctx.Err()
}
func (s *fakeSession) Checkpoint() int { s.mu.Lock(); defer s.mu.Unlock(); return len(s.messages) }
func (s *fakeSession) Restore(mark int) error {
	s.mu.Lock()
	s.messages = s.messages[:mark]
	s.mu.Unlock()
	return nil
}
func (s *fakeSession) Close() error { return nil }

// fakeSynth speaks five seconds of a constant for any text.
type fakeSynth struct{}

func (fakeSynth) SampleRate() int  { return 24000 }
func (fakeSynth) Voices() []string { return nil }
func (fakeSynth) Speak(ctx context.Context, _ speech.SpeakOptions, next func() ([]byte, error), out func([]float32) error) error {
	for {
		if _, err := next(); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return err
		}
	}
	frame := make([]float32, 1920)
	for i := range frame {
		frame[i] = 0.5
	}
	for range 62 {
		if err := out(frame); err != nil {
			return err
		}
	}
	return nil
}
func (fakeSynth) Close() error { return nil }

func speechClip(t *testing.T) []float32 {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, len(raw)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return pcm[:inRate*2] // "And so, my fellow Americans"
}

// run steps the agent in real time through audio, then silence until
// until reports true or the deadline passes.
func run(t *testing.T, c *Cascade, audio []float32, until func(speech.DuplexState, []float32) bool, deadline time.Duration) bool {
	in, out := make([]float32, inFrame), make([]float32, c.outSize)
	tick := time.NewTicker(frameTime)
	defer tick.Stop()
	end := time.Now().Add(deadline)
	for pos := 0; time.Now().Before(end); pos += inFrame {
		<-tick.C
		clear(in)
		if pos+inFrame <= len(audio) {
			copy(in, audio[pos:pos+inFrame])
		}
		state, err := c.Step(context.Background(), in, out)
		if err != nil {
			t.Fatal(err)
		}
		if until(state, out) {
			return true
		}
	}
	return false
}

func TestCascadeAnswersAndStopsWhenInterrupted(t *testing.T) {
	session := &fakeSession{}
	var mu sync.Mutex
	var said []string
	c, err := New(Config{Transcriber: fakeTranscriber{}, TurnDetector: fakeTurns{}, Session: session, Synthesizer: fakeSynth{},
		OnText: func(role chat.Role, text string, final bool) {
			if final {
				mu.Lock()
				said = append(said, text)
				mu.Unlock()
			}
		}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	clip := speechClip(t)
	// Speech, then a pause: the agent answers.
	if !run(t, c, clip, func(s speech.DuplexState, out []float32) bool { return s == speech.Speaking && out[0] == 0.5 }, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	// Speech over the answer, once it is under way, interrupts it: its
	// audio stops.
	run(t, c, nil, func(speech.DuplexState, []float32) bool { return false }, resumeWindow+200*time.Millisecond)
	stopped := run(t, c, clip, func(s speech.DuplexState, out []float32) bool { return s != speech.Speaking }, 2*time.Second)
	if !stopped {
		t.Fatal("the agent kept speaking over the user")
	}
	// Let the responder record the interruption.
	time.Sleep(200 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	session.mu.Lock()
	defer session.mu.Unlock()
	if len(said) < 2 || said[0] != "hello gopher" || !strings.HasSuffix(said[1], "…") {
		t.Fatalf("transcript %q; want the utterance and an interrupted reply", said)
	}
	if session.truncated > len("Hi there, friend.") {
		t.Fatalf("truncated to %d bytes", session.truncated)
	}
}

// Speech just as the answer begins continues the speaker's turn: the answer
// stops and leaves no trace, and the whole utterance is answered once.
func TestCascadeLetsTheSpeakerGoOn(t *testing.T) {
	session := &fakeSession{}
	var mu sync.Mutex
	var said []string
	c, err := New(Config{Transcriber: fakeTranscriber{}, TurnDetector: fakeTurns{}, Session: session, Synthesizer: fakeSynth{},
		OnText: func(role chat.Role, text string, final bool) {
			if final {
				mu.Lock()
				said = append(said, text)
				mu.Unlock()
			}
		}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	clip := speechClip(t)
	speaking := func(s speech.DuplexState, out []float32) bool { return s == speech.Speaking && out[0] == 0.5 }
	if !run(t, c, clip, speaking, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	mu.Lock()
	if len(said) != 0 {
		t.Fatalf("captions %q before the answer was under way", said)
	}
	mu.Unlock()
	// The speaker goes on at once: the answer stops, then comes again.
	if !run(t, c, clip, func(s speech.DuplexState, _ []float32) bool { return s != speech.Speaking }, time.Second) {
		t.Fatal("the agent kept speaking over the speaker going on")
	}
	if !run(t, c, clip[:0], speaking, 5*time.Second) {
		t.Fatal("the agent never answered the whole utterance")
	}
	run(t, c, nil, func(speech.DuplexState, []float32) bool { return false }, resumeWindow+200*time.Millisecond)
	session.mu.Lock()
	defer session.mu.Unlock()
	// The first answer was forgotten; the second is under way.
	if got := strings.Join(session.messages, "|"); got != "hello gopher|Hi there, friend." {
		t.Fatalf("conversation %q", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(said) != 1 || said[0] != "hello gopher" {
		t.Fatalf("captions %q; want the utterance once", said)
	}
}

func TestCascadeStepAllocatesNothing(t *testing.T) {
	c, err := New(Config{Transcriber: fakeTranscriber{}, TurnDetector: fakeTurns{}, Session: &fakeSession{}, Synthesizer: fakeSynth{}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	in, out := make([]float32, inFrame), make([]float32, c.outSize)
	if allocs := testing.AllocsPerRun(100, func() { c.Step(context.Background(), in, out) }); allocs != 0 {
		t.Fatalf("Step allocates %v times", allocs)
	}
}
