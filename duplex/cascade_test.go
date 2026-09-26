// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package duplex

import (
	"context"
	"encoding/binary"
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
	"github.com/GetStream/gophonic/speech/speechtest"
)

// Fake lanes: the cascade's behavior, not the models', is under test. A
// transcriber that hears turns judges each one over; otherwise the turn
// detector does.
type fakeTranscriber struct {
	hears bool
	text  string // what it hears; "hello gopher" if empty
	after int    // samples it needs to make out any word, as a slow speaker's
}

func (f fakeTranscriber) Transcribe(_ context.Context, pcm []float32, opts speech.Options, t *speech.Transcript) error {
	if opts.Turn && !f.hears {
		return speech.ErrUnsupported
	}
	said := f.text
	if said == "" {
		said = "hello gopher"
	}
	if len(pcm) < f.after {
		said = ""
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

func (fakeTurns) Predict([]float32, int, int) (speech.Prediction, error) {
	return speech.Prediction{Probability: 1, Complete: true}, nil
}
func (fakeTurns) Close() error { return nil }

// fakeSession replies from a script, in pieces; without one, "Hi there,
// friend." every time. It records every message.
type fakeSession struct {
	mu        sync.Mutex
	script    []string // replies, in order; a "call:name" reply calls the tool
	messages  []string
	truncated int
	replies   atomic.Int32
	calls     []chat.Call
}

func (s *fakeSession) Add(role chat.Role, text string) error {
	s.mu.Lock()
	s.messages = append(s.messages, text)
	s.mu.Unlock()
	return nil
}

func (s *fakeSession) Reply(ctx context.Context, _ chat.Options, w io.Writer) error {
	s.replies.Add(1)
	s.calls = s.calls[:0]
	reply := "Hi there, friend."
	s.mu.Lock()
	if len(s.script) > 0 {
		reply, s.script = s.script[0], s.script[1:]
	}
	s.mu.Unlock()
	if name, ok := strings.CutPrefix(reply, "call:"); ok {
		s.calls = append(s.calls, chat.Call{Name: name, Arguments: []byte(`{"q": "x"}`)})
		return nil
	}
	s.mu.Lock()
	s.messages = append(s.messages, reply)
	s.mu.Unlock()
	// In pieces of a few bytes, as a model writes.
	for len(reply) > 0 {
		n := min(4, len(reply))
		if _, err := w.Write([]byte(reply[:n])); err != nil {
			return err
		}
		reply = reply[n:]
	}
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
func (s *fakeSession) Calls() []chat.Call { return s.calls }
func (s *fakeSession) Checkpoint() int    { s.mu.Lock(); defer s.mu.Unlock(); return len(s.messages) }
func (s *fakeSession) Restore(mark int) error {
	s.mu.Lock()
	s.messages = s.messages[:mark]
	s.mu.Unlock()
	return nil
}
func (s *fakeSession) Close() error { return nil }

func (s *fakeSession) conversation() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.Join(s.messages, "|")
}

// The fakes' voices speak each byte of a reply for a fixed time: the slow
// one "Hi there, friend." in five seconds, the quick one in a moment.
func slowVoice() *speechtest.Tone  { return speechtest.NewTone(300 * time.Millisecond) }
func quickVoice() *speechtest.Tone { return speechtest.NewTone(25 * time.Millisecond) }

// recorder keeps the final words heard and said, in order.
type recorder struct {
	Base
	mu     sync.Mutex
	final  []string
	voiced []int // Said's voiced counts, in order
}

func (r *recorder) Heard(_ string, text []byte, final bool) {
	if final {
		r.mu.Lock()
		r.final = append(r.final, string(text))
		r.mu.Unlock()
	}
}

func (r *recorder) Said(text []byte, voiced int, final bool) {
	r.mu.Lock()
	r.voiced = append(r.voiced, voiced)
	if final {
		r.final = append(r.final, string(text))
	}
	r.mu.Unlock()
}

func (r *recorder) said() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.final...)
}

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
		state, err := c.Step(in, out)
		if err != nil {
			t.Fatal(err)
		}
		if until(state, out) {
			return true
		}
	}
	return false
}

func speaking(s speech.DuplexState, out []float32) bool  { return s == speech.Speaking && out[0] == 0.5 }
func notSpeaking(s speech.DuplexState, _ []float32) bool { return s != speech.Speaking }
func never(speech.DuplexState, []float32) bool           { return false }

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
	rec := &recorder{}
	c, err := New(Config{Transcriber: asr, Turns: turns, Session: session, Voice: slowVoice(), Observer: rec})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	clip := speechClip(t)
	// Speech, then a pause: the agent answers.
	if !run(t, c, clip, speaking, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	// Speech over the answer, once it is under way, interrupts it: its
	// audio stops.
	run(t, c, nil, never, resumeWindow+200*time.Millisecond)
	if !run(t, c, clip, notSpeaking, 3*time.Second) {
		t.Fatal("the agent kept speaking over the user")
	}
	// Wait for the responder to record the cut reply, rather than for a
	// fixed time a loaded runner may not keep.
	said := rec.said()
	for end := time.Now().Add(3 * time.Second); len(said) < 2 && time.Now().Before(end); said = rec.said() {
		time.Sleep(10 * time.Millisecond)
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if len(said) < 2 || said[0] != "hello gopher" || !strings.HasSuffix(said[1], "…") {
		t.Fatalf("transcript %q; want the utterance and an interrupted reply", said)
	}
	// The conversation keeps what the voice spoke before it was cut, to
	// the piece the model wrote (four bytes, here).
	heard := strings.TrimSuffix(said[1], "…")
	if reply := "Hi there, friend."; !strings.HasPrefix(reply, heard) || len(heard) == len(reply) ||
		session.truncated > len(heard) || session.truncated <= len(heard)-4 {
		t.Fatalf("heard %q, truncated to %d bytes", heard, session.truncated)
	}
}

// Speech over the agent whose words take a while to make out is judged
// again as they arrive: the agent stops once they are heard, not only when
// the talk has gone on for overlapLong.
func TestCascadeJudgesTalkOverAsWordsArrive(t *testing.T) {
	rec := &stages{}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true, after: inRate}, Session: &fakeSession{}, Voice: slowVoice(),
		Interruptions: stopAll{}, Observer: rec})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	clip := speechClip(t)
	if !run(t, c, clip, speaking, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	run(t, c, nil, never, resumeWindow+200*time.Millisecond)
	if !run(t, c, clip, notSpeaking, 3*time.Second) {
		t.Fatal("the agent kept speaking over the user")
	}
	// How much talk it took, not how long: a loaded runner stretches time.
	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.interrupted) != 1 || rec.interrupted[0] < overlapPeek+overlapEvery || rec.interrupted[0] >= overlapLong {
		t.Fatalf("interrupted after %v of talk; want a judgment after words were made out, before %v", rec.interrupted, overlapLong)
	}
}

// stopAll is an interruption judge that stops the agent for any words.
type stopAll struct{}

func (stopAll) Labels() []string { return []string{"go on", "stop"} }
func (stopAll) ClassifyInto(_ context.Context, _ string, probs []float32) error {
	probs[0], probs[1] = 0, 1
	return nil
}
func (stopAll) Close() error { return nil }

// stages records when speech over the agent stopped it.
type stages struct {
	Base
	mu          sync.Mutex
	interrupted []time.Duration
}

func (s *stages) Stage(st Stage, elapsed time.Duration) {
	if st == Interrupted {
		s.mu.Lock()
		s.interrupted = append(s.interrupted, elapsed)
		s.mu.Unlock()
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
	rec := &recorder{}
	c, err := New(Config{Transcriber: asr, Turns: turns, Session: session, Voice: slowVoice(), Observer: rec})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	clip := speechClip(t)
	if !run(t, c, clip, speaking, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	if said := rec.said(); len(said) != 0 {
		t.Fatalf("captions %q before the answer was under way", said)
	}
	// The speaker goes on at once: the answer stops, then comes again.
	if !run(t, c, clip, notSpeaking, time.Second) {
		t.Fatal("the agent kept speaking over the speaker going on")
	}
	if !run(t, c, clip[:0], speaking, 5*time.Second) {
		t.Fatal("the agent never answered the whole utterance")
	}
	run(t, c, nil, never, resumeWindow+200*time.Millisecond)
	// The first answer was forgotten; the second is under way.
	if got := session.conversation(); got != "hello gopher|Hi there, friend." {
		t.Fatalf("conversation %q", got)
	}
	if said := rec.said(); len(said) != 1 || said[0] != "hello gopher" {
		t.Fatalf("captions %q; want the utterance once", said)
	}
}

func TestCascadeStepAllocatesNothing(t *testing.T) {
	c, err := New(Config{Transcriber: fakeTranscriber{}, Turns: fakeTurns{}, Session: &fakeSession{}, Voice: slowVoice()})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	in, out := make([]float32, inFrame), make([]float32, c.outSize)
	if allocs := testing.AllocsPerRun(100, func() { c.Step(in, out) }); allocs != 0 {
		t.Fatalf("Step allocates %v times", allocs)
	}
	if allocs := testing.AllocsPerRun(100, func() { c.Speaker("Ana") }); allocs != 0 {
		t.Fatalf("Speaker allocates %v times", allocs)
	}
}

// A call runs once the turn is over, its result joins the conversation,
// and the agent answers again knowing it.
func TestCascadeRunsTools(t *testing.T) {
	session := &fakeSession{script: []string{"call:lookup"}}
	var runs atomic.Int32
	lookup := chat.Func("lookup", "Looks up.", func(_ context.Context, args struct {
		Q string `json:"q"`
	}) (string, error) {
		runs.Add(1)
		return "found " + args.Q, nil
	})
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Voice: slowVoice(), Tools: []chat.Tool{lookup}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !run(t, c, speechClip(t), speaking, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	if got := session.conversation(); runs.Load() != 1 || got != `hello gopher|found x|Hi there, friend.` {
		t.Fatalf("%d runs; conversation %q", runs.Load(), got)
	}
}

// The model may choose silence: nothing is spoken, what was said joins the
// conversation, and each utterance (and each pause within one) is judged
// again in context.
func TestCascadeKeepsSilence(t *testing.T) {
	session := &fakeSession{script: []string{`<silent until "hi">`, "<silent>", "<silent>", "<silent>", "<silent>", "<silent>"}}
	rec := &recorder{}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Voice: slowVoice(), Observer: rec})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	clip := speechClip(t)
	for i := range 2 {
		if run(t, c, clip, func(s speech.DuplexState, out []float32) bool { return out[0] != 0 }, 3*time.Second) {
			t.Fatalf("the agent spoke in round %d; conversation %q, captions %q", i, session.conversation(), rec.said())
		}
	}
	// The silence's condition is noted before each later moment, until
	// the agent speaks.
	got := session.conversation()
	if session.replies.Load() < 2 || !strings.HasPrefix(got, `hello gopher|<silent until "hi">|You are staying silent until "hi" happens: reply <silent> unless it has.|hello gopher|<silent>`) ||
		strings.Contains(got, "Hi there") {
		t.Fatalf("%d replies; conversation %q", session.replies.Load(), got)
	}
	for _, s := range rec.said() {
		if s != "hello gopher" {
			t.Fatalf("captions %q; want the utterances alone", rec.said())
		}
	}
}

// fakeQuiet judges every message the same way; as a Wake, it says whether
// the message ends a silence.
type fakeQuiet struct{ asks bool }

func (fakeQuiet) Labels() []string { return quietLabels }
func (q fakeQuiet) ClassifyInto(_ context.Context, _ string, probs []float32) error {
	probs[0], probs[1] = 1, 0
	if q.asks {
		probs[0], probs[1] = 0, 1
	}
	return nil
}
func (fakeQuiet) Close() error { return nil }

// With a Wake, a silence the model chose until something happens is kept
// through words that do not end it, without asking the model, and ended
// by words that do, the model told so.
func TestCascadeWakes(t *testing.T) {
	session := &fakeSession{script: []string{`<silent until "hi">`, "Hello!"}}
	rec := &recorder{}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Voice: quickVoice(),
		Wake: fakeQuiet{asks: false}, Observer: rec})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	clip := speechClip(t)
	for range 2 {
		if run(t, c, clip, speaking, 3*time.Second) {
			t.Fatalf("the agent spoke; conversation %q", session.conversation())
		}
	}
	if got := session.conversation(); session.replies.Load() != 1 || !strings.HasPrefix(got, `hello gopher|<silent until "hi">|hello gopher`) {
		t.Fatalf("%d replies; conversation %q", session.replies.Load(), got)
	}
	c.cfg.Wake = fakeQuiet{asks: true}
	if !run(t, c, clip, speaking, 5*time.Second) {
		t.Fatalf("the agent stayed silent; conversation %q", session.conversation())
	}
	if got := session.conversation(); !strings.HasSuffix(got, `What you were waiting for ("hi") has happened: answer now.|hello gopher|Hello!`) {
		t.Fatalf("conversation %q", got)
	}
}

// A silence until something happens, chosen when no one asked for quiet,
// is overruled: the agent answers, told why.
func TestCascadeOverrulesSilence(t *testing.T) {
	session := &fakeSession{script: []string{`<silent until "return">`}}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true, text: "Return."}, Session: session, Voice: quickVoice(),
		Quiet: fakeQuiet{asks: false}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !run(t, c, speechClip(t), speaking, 5*time.Second) {
		t.Fatal("the agent kept a silence no one asked for")
	}
	run(t, c, nil, notSpeaking, 3*time.Second)
	if got := session.conversation(); !strings.HasSuffix(got, "Return.|"+notAskedQuiet+"|Hi there, friend.") || strings.Contains(got, "<silent") {
		t.Fatalf("conversation %q", got)
	}
	// Asked for, the silence stands.
	session = &fakeSession{script: []string{`<silent until "hi">`, "<silent>", "<silent>", "<silent>"}}
	c2, err := New(Config{Transcriber: fakeTranscriber{hears: true, text: "Be quiet until I say hi."}, Session: session, Voice: quickVoice(),
		Quiet: fakeQuiet{asks: true}})
	if err != nil {
		t.Fatal(err)
	}
	defer c2.Close()
	if run(t, c2, speechClip(t), speaking, 3*time.Second) {
		t.Fatalf("the agent spoke; conversation %q", session.conversation())
	}
}

// A reply that ends with <silent 1s> is spoken, and the agent is asked
// again after a second: a reminder needs no tool.
func TestCascadePlansAMoment(t *testing.T) {
	session := &fakeSession{script: []string{"Sure, I will remind you. <silent 1s>", "This is your reminder."}}
	rec := &recorder{}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Voice: quickVoice(), Observer: rec})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !run(t, c, speechClip(t), speaking, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	if !run(t, c, nil, notSpeaking, 3*time.Second) {
		t.Fatal("the agent never finished")
	}
	began := time.Now()
	if !run(t, c, nil, speaking, 4*time.Second) {
		t.Fatal("the agent never came back")
	}
	if since := time.Since(began); since < 800*time.Millisecond {
		t.Fatalf("came back after %v; want about a second", since)
	}
	run(t, c, nil, notSpeaking, 3*time.Second)
	time.Sleep(100 * time.Millisecond)
	if got := session.conversation(); got != "hello gopher|Sure, I will remind you. <silent 1s>|The time you asked to wait has passed.|This is your reminder." {
		t.Fatalf("conversation %q", got)
	}
	if said := rec.said(); len(said) != 3 || said[1] != "Sure, I will remind you." || said[2] != "This is your reminder." {
		t.Fatalf("captions %q", said)
	}
}

// A note is a moment: the agent may answer it, once no one is speaking.
func TestCascadeAnswersNotes(t *testing.T) {
	session := &fakeSession{script: []string{"Hi Ana!"}}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Voice: quickVoice()})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if err := c.Note("Ana joined the call."); err != nil {
		t.Fatal(err)
	}
	if !run(t, c, nil, speaking, 3*time.Second) {
		t.Fatal("the agent never answered the note")
	}
	run(t, c, nil, notSpeaking, 3*time.Second)
	if got := session.conversation(); got != "Ana joined the call.|Hi Ana!" {
		t.Fatalf("conversation %q", got)
	}
}

// After Idle of quiet the agent is asked once whether it has something to
// say.
func TestCascadeIdleMoment(t *testing.T) {
	session := &fakeSession{script: []string{"Is anyone there?"}}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Voice: quickVoice(), Idle: 300 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !run(t, c, nil, speaking, 3*time.Second) {
		t.Fatal("the agent never spoke up")
	}
	run(t, c, nil, notSpeaking, 3*time.Second)
	if got := session.conversation(); got != "Nothing has happened for a while.|Is anyone there?" {
		t.Fatalf("conversation %q", got)
	}
}

// The speaker's name joins the conversation once, when the speaker
// changes, and the words stay as they were said.
func TestCascadeNotesTheSpeaker(t *testing.T) {
	session := &fakeSession{}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Voice: quickVoice()})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.Speaker("Ana")
	clip := speechClip(t)
	for range 2 {
		if !run(t, c, clip, speaking, 5*time.Second) {
			t.Fatal("the agent never spoke")
		}
		run(t, c, nil, notSpeaking, 3*time.Second)
	}
	if got := session.conversation(); got != "Ana is speaking.|hello gopher|Hi there, friend.|hello gopher|Hi there, friend." {
		t.Fatalf("conversation %q", got)
	}
}

func TestMarker(t *testing.T) {
	for _, c := range []struct {
		in    string
		n     int
		wait  time.Duration
		until string
		ok    bool
	}{
		{"<silent>", 8, 0, "", true},
		{"<silent> and more", 8, 0, "", true},
		{"<silent 30s>", 12, 30 * time.Second, "", true},
		{"<silent 30>", 11, 30 * time.Second, "", true},
		{"<silent for 2 minutes>", 22, 2 * time.Minute, "", true},
		{"<silent 500ms>", 14, 500 * time.Second, "", true}, // no milliseconds: seconds
		{`<silent until "hi">`, 19, 0, "hi", true},
		{`<silent until Ana says go>`, 26, 0, "Ana says go", true},
		{"<silent until>", 14, 0, "being asked to speak", true},
		{"<sil", -1, 0, "", false},
		{"<silent 30", -1, 0, "", false},
		{"<3 you", 0, 0, "", false},
		{"hello", 0, 0, "", false},
	} {
		n, wait, until, ok := marker([]byte(c.in))
		if n != c.n || wait != c.wait || until != c.until || ok != c.ok {
			t.Errorf("%q: %d %v %q %v; want %d %v %q %v", c.in, n, wait, until, ok, c.n, c.wait, c.until, c.ok)
		}
	}
}

func TestEchoOf(t *testing.T) {
	for _, c := range []struct {
		reply, message string
		n              int // bytes of reply that repeat, if it does
		holding        bool
	}{
		{"Three, two, one. Got it.", "Three, two, one.", len("Three, two, one"), false},
		{"How is it going? I'm well.", "How is it going?", len("How is it going"), false},
		{"How is", "How is it going?", 0, true},
		{"How is it g", "How is it going?", 0, true},
		{"How about you?", "How is it going?", 0, false},
		{"Sempre é só fazer tudo bem.", "sempre é só fazer tudo bem.", len("Sempre é só fazer tudo bem"), false},
		{"So, you're heading out!", "How is it going?", 0, false},
		{"Hi there.", "Hi.", 0, false}, // short messages are answered, not repeated
	} {
		n, holding := echoOf(c.reply, c.message)
		if n != c.n || holding != c.holding {
			t.Errorf("%q: %d, %v; want %d, %v", c.reply, n, holding, c.n, c.holding)
		}
	}
}

// A reply that reads back what it answers says only the rest, and the
// conversation keeps it as said.
func TestCascadeDropsEcho(t *testing.T) {
	session := &fakeSession{script: []string{"How is it going? Fine, thanks."}}
	rec := &recorder{}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true, text: "How is it going?"}, Session: session, Voice: slowVoice(), Observer: rec})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !run(t, c, speechClip(t), func(s speech.DuplexState, _ []float32) bool { return s == speech.Speaking }, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	run(t, c, nil, func(s speech.DuplexState, _ []float32) bool { return s == speech.Listening }, 6*time.Second)
	time.Sleep(100 * time.Millisecond)
	said := rec.said()
	if got := session.conversation(); len(said) < 2 || said[1] != "Fine, thanks." || !strings.HasSuffix(got, "|Fine, thanks.") {
		t.Fatalf("said %q; conversation %q", said, got)
	}
}

func TestWordEnd(t *testing.T) {
	for _, c := range []struct {
		s    string
		n    int
		want int
	}{
		{"Hi there, friend.", 0, 0},
		{"Hi there, friend.", 2, 2},
		{"Hi there, friend.", 3, 3},
		{"Hi there, friend.", 6, 3}, // "the" of "there": back to "Hi "
		{"Hi there, friend.", 8, 8}, // "there", before its comma
		{"Hi there, friend.", 17, 17},
		{"It's three o'clock.", 13, 11}, // within "o'clock"
		{"Olá, você está bem?", 9, 6},   // within "você"
		{"今天天气很好", 6, 6},                // one character in, a whole word
		{"今天天气很好", 7, 6},                // within a character's bytes
		{"Photosynthesis", 5, 0},
	} {
		if got := wordEnd([]byte(c.s), c.n); got != c.want {
			t.Errorf("wordEnd(%q, %d) = %d, want %d", c.s, c.n, got, c.want)
		}
	}
}

// Captions follow the voice: the reply shows a word at a time as the
// voice speaks it, never half of one, and whole once it is heard.
func TestCascadeCaptionsFollowTheVoice(t *testing.T) {
	session := &fakeSession{}
	rec := &recorder{}
	c, err := New(Config{Transcriber: fakeTranscriber{hears: true}, Session: session, Voice: speechtest.NewTone(120 * time.Millisecond), Observer: rec})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !run(t, c, speechClip(t), speaking, 5*time.Second) {
		t.Fatal("the agent never spoke")
	}
	run(t, c, nil, notSpeaking, 5*time.Second)
	time.Sleep(100 * time.Millisecond)
	reply := "Hi there, friend."
	rec.mu.Lock()
	defer rec.mu.Unlock()
	shown, last := map[int]bool{}, 0
	for _, v := range rec.voiced {
		if v < last || v > len(reply) || wordEnd([]byte(reply), v) != v {
			t.Fatalf("captions at %v of %q", rec.voiced, reply)
		}
		last = v
		shown[v] = true
	}
	if last != len(reply) || len(shown) < 4 {
		t.Fatalf("captions at %v of %q: want it word by word to the end", rec.voiced, reply)
	}
}

func (s *fakeSession) AddCalls(text string, _ []chat.Call) error { return s.Add(chat.Assistant, text) }

// interrupt cancels the reply before it empties playback: a tick can find
// playback empty without having seen the cancellation, and the reply is
// cut all the same, never heard to its end.
func TestPlaybackDrainKeepsTheCut(t *testing.T) {
	for _, cut := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		c := &Cascade{play: newRing[float32](1), cancel: cancel, tick: time.NewTicker(time.Millisecond)}
		c.play.write([]float32{0.5})
		ticks := 0
		got := c.waitPlayback(ctx, func() {
			ticks++
			if cut {
				c.interrupt()
			} else {
				var heard [1]float32
				c.play.read(heard[:])
			}
		})
		c.tick.Stop()
		cancel()
		if ticks != 1 || got != cut {
			t.Fatalf("cut %v: %d ticks, reported cut %v", cut, ticks, got)
		}
	}
	// Cut before the wait begins, with playback already emptied.
	ctx, cancel := context.WithCancel(context.Background())
	c := &Cascade{play: newRing[float32](1), cancel: cancel, tick: time.NewTicker(time.Millisecond)}
	defer c.tick.Stop()
	c.play.write([]float32{0.5})
	c.interrupt()
	if !c.waitPlayback(ctx, func() { t.Fatal("a caption after the cut") }) {
		t.Fatal("a cut reply reported heard")
	}
}
