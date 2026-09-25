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
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/speech"
)

// Fake lanes: the cascade's behavior, not the models', is under test. A
// transcriber that hears turns judges each one over; otherwise the turn
// detector does.
type fakeTranscriber struct {
	hears bool
	text  string // what it hears; "hello gopher" if empty
}

func (f fakeTranscriber) Transcribe(_ context.Context, _ []float32, opts speech.Options, t *speech.Transcript) error {
	if opts.Turn && !f.hears {
		return speech.ErrUnsupported
	}
	said := f.text
	if said == "" {
		said = "hello gopher"
	}
	t.Text = append(t.Text[:0], said...)
	t.Turn = speech.Prediction{}
	if opts.Turn {
		t.Turn = speech.Prediction{Probability: 0.95, Complete: true}
	}
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
func (s *fakeSession) Calls() []chat.Call { return nil }
func (s *fakeSession) Checkpoint() int    { s.mu.Lock(); defer s.mu.Unlock(); return len(s.messages) }
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
	said := 0
	for {
		p, err := next()
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return err
		}
		said += len(p)
	}
	if said == 0 {
		return nil // nothing to say
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

// Both ways of judging turns: a turn detector, or the transcriber itself.
var judges = []struct {
	name  string
	asr   fakeTranscriber
	turns speech.TurnDetector
}{{"turn detector", fakeTranscriber{}, fakeTurns{}}, {"transcriber", fakeTranscriber{hears: true}, nil}}

func TestCascadeAnswersAndStopsWhenInterrupted(t *testing.T) {
	for _, j := range judges {
		t.Run(j.name, func(t *testing.T) { answersAndStops(t, j.asr, j.turns) })
	}
}

func answersAndStops(t *testing.T, asr fakeTranscriber, turns speech.TurnDetector) {
	session := &fakeSession{}
	var mu sync.Mutex
	var said []string
	c, err := New(Config{Transcriber: asr, TurnDetector: turns, Session: session, Synthesizer: fakeSynth{},
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
	for _, j := range judges {
		t.Run(j.name, func(t *testing.T) { letsTheSpeakerGoOn(t, j.asr, j.turns) })
	}
}

func letsTheSpeakerGoOn(t *testing.T, asr fakeTranscriber, turns speech.TurnDetector) {
	session := &fakeSession{}
	var mu sync.Mutex
	var said []string
	c, err := New(Config{Transcriber: asr, TurnDetector: turns, Session: session, Synthesizer: fakeSynth{},
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

// toolSession answers a user's message with a call to lookup, then, given
// its result, in words.
type toolSession struct {
	fakeSession
	calls []chat.Call
}

func (s *toolSession) Reply(ctx context.Context, opts chat.Options, sink func([]byte) error) error {
	s.mu.Lock()
	last := s.messages[len(s.messages)-1]
	s.mu.Unlock()
	s.calls = s.calls[:0]
	if last == "hello gopher" {
		s.calls = append(s.calls, chat.Call{Name: "lookup", Arguments: []byte(`{"q": "x"}`)})
		return nil
	}
	return s.fakeSession.Reply(ctx, opts, sink)
}

func (s *toolSession) Calls() []chat.Call { return s.calls }

// A call runs once the turn is over, its result joins the conversation,
// and a result that needs words has the agent answer.
func TestCascadeRunsTools(t *testing.T) {
	session := &toolSession{}
	var runs atomic.Int32
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Synthesizer: fakeSynth{},
		Tools: []Tool{{ToolSpec: chat.ToolSpec{Name: "lookup"}, Run: func(_ context.Context, args []byte) (string, bool, error) {
			runs.Add(1)
			return "found " + string(args), true, nil
		}}}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !run(t, c, speechClip(t), func(s speech.DuplexState, out []float32) bool { return s == speech.Speaking && out[0] == 0.5 }, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if got := strings.Join(session.messages, "|"); runs.Load() != 1 || got != `hello gopher|found {"q": "x"}|Hi there, friend.` {
		t.Fatalf("%d runs; conversation %q", runs.Load(), got)
	}
}

// silentSession always chooses silence until "hi", in pieces.
type silentSession struct {
	fakeSession
	replies atomic.Int32
}

func (s *silentSession) Reply(ctx context.Context, _ chat.Options, sink func([]byte) error) error {
	s.replies.Add(1)
	for _, p := range []string{"<sil", "ent until ", `"hi">`} {
		if err := sink([]byte(p)); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.messages = append(s.messages, `<silent until "hi">`)
	s.mu.Unlock()
	return nil
}

type fakeWake struct{ speak float32 }

func (w fakeWake) Labels() []string { return []string{"stay", "speak"} }
func (w fakeWake) ClassifyInto(_ context.Context, _ string, probs []float32) error {
	probs[0], probs[1] = 1-w.speak, w.speak
	return nil
}
func (fakeWake) Close() error { return nil }

// The model may choose silence: nothing is spoken, and until what it named
// happens, what is said joins the conversation without being answered.
func TestCascadeKeepsSilence(t *testing.T) {
	session := &silentSession{}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Synthesizer: fakeSynth{}, Wake: fakeWake{speak: 0.1}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	clip := speechClip(t)
	for range 2 {
		if run(t, c, clip, func(s speech.DuplexState, out []float32) bool { return out[0] != 0 }, 3*time.Second) {
			t.Fatal("the agent spoke")
		}
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	// One reply, the silence; what was said after it is context only.
	got := strings.Join(session.messages, "|")
	if session.replies.Load() != 1 || !strings.HasPrefix(got, `hello gopher|<silent until "hi">|hello gopher`) ||
		strings.Count(got, "<silent") != 1 {
		t.Fatalf("%d replies; conversation %q", session.replies.Load(), got)
	}
}

func TestEchoOf(t *testing.T) {
	for _, c := range []struct {
		reply   string
		n       int // bytes of reply that repeat, if it does
		holding bool
	}{
		{"Three, two, one. Got it.", len("Three, two, one"), false},
		{"Cherry Judge said: How is it going? Fine.", len("Cherry Judge said: How is it going"), false},
		{"How is it going? I'm well.", len("How is it going"), false},
		{"How is", 0, true},
		{"How is it g", 0, true},
		{"How about you?", 0, false},
		{"Sempre é só fazer tudo bem.", len("Sempre é só fazer tudo bem"), false},
		{"So, you're heading out!", 0, false},
	} {
		n, holding := echoOf(c.reply, "Cherry Judge said: How is it going?", "How is it going?")
		if c.reply[0] == 'T' {
			n, holding = echoOf(c.reply, "Three, two, one.")
		}
		if c.reply[0] == 'S' {
			n, holding = echoOf(c.reply, "sempre é só fazer tudo bem.")
		}
		if n != c.n || holding != c.holding {
			t.Errorf("%q: %d, %v; want %d, %v", c.reply, n, holding, c.n, c.holding)
		}
	}
}

// echoSession reads the question back before answering it.
type echoSession struct{ fakeSession }

func (s *echoSession) Reply(ctx context.Context, _ chat.Options, sink func([]byte) error) error {
	for _, p := range []string{"How is it ", "going? ", "Fine, ", "thanks."} {
		if err := sink([]byte(p)); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.messages = append(s.messages, "How is it going? Fine, thanks.")
	s.mu.Unlock()
	return nil
}

// A reply that reads back what it answers says only the rest, and the
// conversation keeps it as said.
func TestCascadeDropsEcho(t *testing.T) {
	session := &echoSession{}
	var mu sync.Mutex
	var said []string
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true, text: "How is it going?"}, Session: session, Synthesizer: fakeSynth{},
		OnText: func(role chat.Role, text string, final bool) {
			if final && role == chat.Assistant {
				mu.Lock()
				said = append(said, text)
				mu.Unlock()
			}
		}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !run(t, c, speechClip(t), func(s speech.DuplexState, out []float32) bool { return s == speech.Speaking }, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	run(t, c, nil, func(s speech.DuplexState, _ []float32) bool { return s == speech.Listening }, 5*time.Second)
	time.Sleep(100 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	session.mu.Lock()
	defer session.mu.Unlock()
	if len(said) == 0 || said[0] != "Fine, thanks." || session.messages[len(session.messages)-1] != "Fine, thanks." {
		t.Fatalf("said %q; conversation %q", said, session.messages)
	}
}
