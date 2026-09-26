// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package scenario

import (
	"context"
	"io"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/speech"
	"github.com/GetStream/gophonic/speech/speechtest"
)

// echoDuplex is a speech.Duplex that, a moment after hearing a second of
// sound, speaks a second of tone unless it was told to be quiet; a note
// makes it speak too.
type echoDuplex struct {
	mu      sync.Mutex
	heard   int  // frames of sound heard in a row
	quiet   bool // told to be quiet
	pending int  // frames of tone left to speak
	silence int  // frames since the sound ended
	notes   []string
}

func (d *echoDuplex) Rates() (int, int) { return 16000, 24000 }
func (d *echoDuplex) Frame() (int, int) { return 320, 480 }
func (d *echoDuplex) Step(in, out []float32) (speech.DuplexState, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	var e float32
	for _, v := range in {
		e += v * v
	}
	if e > 0.01 {
		d.heard++
		d.silence = 0
	} else if d.heard > 0 {
		d.silence++
		if d.silence == 10 { // 200 ms of pause
			if d.heard >= 25 && !d.quiet {
				d.pending = 50
			}
			d.heard = 0
		}
	}
	clear(out)
	if d.pending > 0 {
		for i := range out {
			out[i] = 0.3
		}
		d.pending--
		return speech.Speaking, nil
	}
	return speech.Listening, nil
}
func (d *echoDuplex) Say(string) error { return nil }
func (d *echoDuplex) Note(text string) error {
	d.mu.Lock()
	d.notes = append(d.notes, text)
	d.quiet = strings.Contains(text, "quiet")
	if !d.quiet {
		d.pending = 25
	}
	d.mu.Unlock()
	return nil
}
func (d *echoDuplex) Close() error { return nil }

// fixedEars hears the same words in any sound.
type fixedEars struct{ text string }

func (e fixedEars) Transcribe(_ context.Context, pcm []float32, _ speech.Options, t *speech.Transcript) error {
	t.Reset()
	if len(pcm) > 0 {
		t.Text = append(t.Text, e.text...)
	}
	return nil
}
func (fixedEars) Close() error { return nil }

// yesJudge says yes when the claim, which ends the question, mentions a
// tone.
type yesJudge struct{}

func (yesJudge) NewSession(string, ...chat.ToolSpec) (chat.Session, error) { return &yesSession{}, nil }
func (yesJudge) Close() error                                              { return nil }

type yesSession struct{ asked string }

func (s *yesSession) Add(_ chat.Role, text string) error { s.asked = text; return nil }
func (s *yesSession) Reply(_ context.Context, _ chat.Options, w io.Writer) error {
	if strings.Contains(s.asked, "tone?") {
		_, err := w.Write([]byte("Yes."))
		return err
	}
	_, err := w.Write([]byte("No."))
	return err
}
func (*yesSession) Calls() []chat.Call                                           { return nil }
func (*yesSession) Finished(context.Context, chat.Role, string) (float32, error) { return 1, nil }
func (*yesSession) Prefill(context.Context) error                                { return nil }
func (*yesSession) Truncate(int) error                                           { return nil }
func (*yesSession) Checkpoint() int                                              { return 0 }
func (*yesSession) Restore(int) error                                            { return nil }
func (*yesSession) Close() error                                                 { return nil }

func TestParse(t *testing.T) {
	lines, expected, err := parse(`# expect: fail
# a comment
user: Hello there.
gopher: speaks within 3s
gopher(pt): says that it is a tone
user(pt): Olá.
gopher: silent for 1s
gopher: does not repeat the user
gopher: captions follow the voice
wait: 250ms
chat: Ana: see you at 3
join: Ana
note: something happened
`)
	if err != nil || expected != "fail" {
		t.Fatalf("%v, expected %q", err, expected)
	}
	want := []line{
		{kind: "user", arg: "Hello there."},
		{kind: "speaks", dur: 3 * time.Second},
		{kind: "says", arg: "that it is a tone", lang: speech.Portuguese},
		{kind: "user", arg: "Olá.", lang: speech.Portuguese},
		{kind: "silent", dur: time.Second},
		{kind: "repeats"},
		{kind: "captions"},
		{kind: "wait", dur: 250 * time.Millisecond},
		{kind: "chat", arg: "Ana: see you at 3"},
		{kind: "join", arg: "Ana"},
		{kind: "note", arg: "something happened"},
	}
	if len(lines) != len(want) {
		t.Fatalf("%d lines, want %d", len(lines), len(want))
	}
	for i, l := range lines {
		l.text = ""
		if l != want[i] {
			t.Errorf("line %d: %+v, want %+v", i+1, l, want[i])
		}
	}
	if _, _, err := parse("gopher: dances"); err == nil {
		t.Fatal("an unknown assertion parsed")
	}
}

// The harness speaks to the agent, hears its answer, judges it, and
// reports silence and failures as they are.
func TestRun(t *testing.T) {
	agent := &echoDuplex{}
	cfg := Config{Agent: agent, Voice: speechtest.NewTone(80 * time.Millisecond), Ears: fixedEars{"a tone"}, Judge: yesJudge{},
		Answer: 3 * time.Second, Quiet: time.Second, Log: t.Logf}
	report, err := Run(context.Background(), cfg, `
user: Say something.
gopher: speaks
user: Say it again.
gopher: says that it is a tone
gopher: does not repeat the user
note: please be quiet
user: Anything?
gopher: silent
note: speak up
gopher: speaks within 2s
gopher: says that it is a song
`)
	if err != nil {
		t.Fatal(err)
	}
	var got []bool
	for _, s := range report.Steps {
		got = append(got, s.OK)
	}
	want := []bool{true, true, true, true, true, true, true, true, true, true, false}
	if len(got) != len(want) {
		t.Fatalf("%d steps: %v", len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("step %d %q: ok %v, want %v (%s)", i+1, report.Steps[i].Line, got[i], want[i], report.Steps[i].Reason)
		}
	}
	if s := report.Steps[1]; s.Heard != "a tone" || s.After <= 0 || s.After > 2*time.Second {
		t.Errorf("speaks: heard %q after %v", s.Heard, s.After)
	}
	if agent.notes[0] != "please be quiet" || agent.notes[1] != "speak up" {
		t.Errorf("notes %q", agent.notes)
	}
}

func TestWordError(t *testing.T) {
	for _, c := range []struct {
		said, heard string
		want        float64
	}{
		{"Gopher, stop talking until I say hi.", "Gopher stop talking until I say hi", 0},
		{"Gopher, qual é a capital de Portugal?", "Governo é a capital de Portugal.", 2.0 / 7},
		{"Hi!", "", 1},
		{"", "anything", 0},
	} {
		if got := wordError(c.said, c.heard); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("wordError(%q, %q) = %v, want %v", c.said, c.heard, got, c.want)
		}
	}
}
