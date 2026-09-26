// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package scenario tests any speech.Duplex end to end from a text script,
// with real audio: the user's lines are spoken by a synthesizer, the
// agent's speech is transcribed, and a local language model judges what it
// said. A script is a dialogue, one step per line:
//
//	user: Stop talking until I say hi.
//	gopher: silent
//	user: What is the capital of Portugal?
//	gopher: silent
//	user: Hi!
//	gopher: speaks
//	user: What is the capital of Portugal?
//	gopher: says that the capital is Lisbon
//
// A line that starts with the user speaks; wait, note, chat, join, leave,
// and say are directives; any other name is the agent, whose line is an
// assertion: silent [for 20s], speaks [within 40s], says <claim>, does
// not repeat the user, captions follow the voice. Talk over the agent is
// scripted with starts [within 10s], which waits for the agent to begin a
// reply and not for its end, then a user line: stops within 1.5s holds if
// the words stopped the agent within that long of their start, and goes
// on if the reply went on to its end. user(pt) speaks
// Portuguese, and gopher(pt) is transcribed as Portuguese. A script whose
// first line is "# expect: fail" documents a behavior that does not work
// yet: its failures are reported, not counted. Scenarios run in real time,
// since an agent's turn taking follows the clock.
package scenario

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/duplex"
	"github.com/GetStream/gophonic/speech"
)

// Config is what a scenario runs with.
type Config struct {
	// Agent is under test.
	Agent speech.Duplex
	// Voice speaks the user's lines, as Speak says; a voice other than
	// the agent's makes the transcripts easier to tell apart.
	Voice speech.Synthesizer
	Speak speech.SpeakOptions
	// Ears transcribes the agent's speech, as Listen says (the languages
	// of the call, as the agent listens).
	Ears   speech.Transcriber
	Listen speech.Options
	// Judge decides whether the agent said what a "says" line claims; nil
	// fails such lines.
	Judge chat.Generator
	// Captions, when the agent was built with it as its Observer, lets
	// "captions follow the voice" be checked.
	Captions *Captions
	// Answer is how long the agent has to start speaking (default 8 s);
	// Quiet how long its silence is watched (default 6 s).
	Answer, Quiet time.Duration
	// Log, when not nil, receives each step's outcome.
	Log func(format string, args ...any)
}

// Report is the outcome of a scenario: one Step per line.
type Report struct {
	Steps []Step
	// Expected reports the script's expectation: "fail" for a behavior
	// that does not work yet.
	Expected string
}

// Step is one line's outcome.
type Step struct {
	Line    string
	Heard   string        // what the agent said, transcribed, for an assertion
	After   time.Duration // from the end of the user's line to the agent's first sound
	Stopped time.Duration // for "stops": from the user's first sound to the agent stopping
	OK      bool
	Reason  string
}

// OK reports whether every step passed.
func (r Report) OK() bool {
	for _, s := range r.Steps {
		if !s.OK {
			return false
		}
	}
	return true
}

// Captions is a duplex.Observer that records the agent's captions, for
// "captions follow the voice", and logs the reply's stages and errors
// through Log, when set (Test sets it).
type Captions struct {
	duplex.Base
	Log         func(format string, args ...any)
	mu          sync.Mutex
	said        []caption
	interrupted []time.Time // when speech over the agent stopped it
}

func (c *Captions) Stage(s duplex.Stage, elapsed time.Duration) {
	if s == duplex.Interrupted {
		c.mu.Lock()
		c.interrupted = append(c.interrupted, time.Now())
		c.mu.Unlock()
	}
	if c.Log != nil {
		c.Log("stage %s after %v", s, elapsed.Round(time.Millisecond))
	}
}

// interruptedSince returns when speech over the agent first stopped it at
// or after t.
func (c *Captions) interruptedSince(t time.Time) (time.Time, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, at := range c.interrupted {
		if !at.Before(t) {
			return at, true
		}
	}
	return time.Time{}, false
}

func (c *Captions) Error(err error) {
	if c.Log != nil {
		c.Log("agent error: %v", err)
	}
}

type caption struct {
	text   string
	voiced int
	final  bool
	at     time.Time
}

func (c *Captions) Said(text []byte, voiced int, final bool) {
	c.mu.Lock()
	c.said = append(c.said, caption{string(text), voiced, final, time.Now()})
	c.mu.Unlock()
}

// since returns the captions recorded since t.
func (c *Captions) since(t time.Time) []caption {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []caption
	for _, s := range c.said {
		if !s.at.Before(t) {
			out = append(out, s)
		}
	}
	return out
}

// Test runs every script matching glob as a subtest of t. setup makes
// each script's Config, with a fresh agent, so that no conversation leaks
// from one script into the next; it registers what it opens with
// t.Cleanup.
func Test(t *testing.T, setup func(t *testing.T) Config, glob string) {
	t.Helper()
	files, err := filepath.Glob(glob)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatalf("no scenarios match %s", glob)
	}
	for _, file := range files {
		t.Run(strings.TrimSuffix(filepath.Base(file), filepath.Ext(file)), func(t *testing.T) {
			script, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			c := setup(t)
			c.Log = t.Logf
			if c.Captions != nil {
				c.Captions.Log = t.Logf
			}
			report, err := Run(t.Context(), c, string(script))
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range report.Steps {
				if !s.OK {
					if report.Expected == "fail" {
						t.Logf("known failure: %s: %s", s.Line, s.Reason)
					} else {
						t.Errorf("%s: %s", s.Line, s.Reason)
					}
				}
			}
			if report.Expected == "fail" && report.OK() {
				t.Logf("the scenario passes now: drop its expect: fail line")
			}
		})
	}
}

// Run drives cfg.Agent through script and reports each step's outcome. It
// returns an error only when the harness itself fails: a lane error, or a
// line it cannot read.
func Run(ctx context.Context, cfg Config, script string) (Report, error) {
	if cfg.Answer == 0 {
		cfg.Answer = 12 * time.Second
	}
	if cfg.Quiet == 0 {
		cfg.Quiet = 6 * time.Second
	}
	if cfg.Log == nil {
		cfg.Log = func(string, ...any) {}
	}
	lines, expected, err := parse(script)
	if err != nil {
		return Report{}, err
	}
	r := newRunner(ctx, cfg)
	defer r.stop()
	report := Report{Expected: expected}
	for _, l := range lines {
		step, err := r.step(l)
		if err != nil {
			return report, fmt.Errorf("%s: %w", l.text, err)
		}
		if step.OK {
			cfg.Log("ok   %s%s", l.text, step.detail())
		} else {
			cfg.Log("FAIL %s: %s%s", l.text, step.Reason, step.detail())
		}
		report.Steps = append(report.Steps, step)
	}
	return report, nil
}

func (s Step) detail() string {
	var b strings.Builder
	if s.Stopped > 0 {
		fmt.Fprintf(&b, " (after %v)", s.Stopped.Round(10*time.Millisecond))
	}
	if s.Heard != "" {
		fmt.Fprintf(&b, " (heard %q", s.Heard)
		if s.After > 0 {
			fmt.Fprintf(&b, " after %v", s.After.Round(10*time.Millisecond))
		}
		b.WriteString(")")
	}
	return b.String()
}

// runner steps the agent on its own goroutine, on a 20 ms clock, feeding
// it the user's audio and collecting what it speaks.
type runner struct {
	cfg                                Config
	ctx                                context.Context
	cancel                             context.CancelFunc
	done                               chan struct{}
	inRate, outRate, inFrame, outFrame int

	mu         sync.Mutex
	queue      []float32 // the user's audio, at the agent's input rate
	spoke      []float32 // the agent's audio since the last collect
	speaking   bool
	state      speech.DuplexState
	lastSound  time.Time
	firstSound time.Time // since the last user line
	userEnded  time.Time // when the user's last audio was fed
	userLead   int       // samples of silence before the user's line sounds
	userFed    int       // samples of the user's line fed so far
	userBegan  time.Time // when the user's line began to sound
	quietAt    time.Time // when the agent last stopped speaking
	quietMark  int       // len(spoke) then
	resumedAt  time.Time // the agent's first sound after quietAt
	resampler  *speech.Resampler
	lastUser   string
	speaker    func(string)

	// The reply under examination: assertions that follow one another
	// without a new stimulus describe the same reply.
	fresh      bool // a stimulus happened since the last reply was captured
	stimulusAt time.Time
	lastPCM    []float32
	lastHeard  string
	lastLang   speech.Language
	lastAfter  time.Duration
}

// stimulus notes that something new happened: the next assertion waits
// for a new reply, and what the agent said before is not part of it.
func (r *runner) stimulus() {
	r.fresh, r.stimulusAt = true, time.Now()
	r.mu.Lock()
	r.spoke, r.firstSound = r.spoke[:0], time.Time{}
	r.mu.Unlock()
}

func newRunner(ctx context.Context, cfg Config) *runner {
	ctx, cancel := context.WithCancel(ctx)
	r := &runner{cfg: cfg, ctx: ctx, cancel: cancel, done: make(chan struct{}), resampler: speech.NewResampler()}
	r.inRate, r.outRate = cfg.Agent.Rates()
	r.inFrame, r.outFrame = cfg.Agent.Frame()
	if s, ok := cfg.Agent.(interface{ Speaker(string) }); ok {
		r.speaker = s.Speaker
	}
	go r.loop()
	return r
}

func (r *runner) stop() {
	r.cancel()
	<-r.done
}

// loop is the agent's clock.
func (r *runner) loop() {
	defer close(r.done)
	in, out := make([]float32, r.inFrame), make([]float32, r.outFrame)
	tick := time.NewTicker(time.Second / time.Duration(r.inRate/r.inFrame))
	defer tick.Stop()
	for {
		select {
		case <-r.ctx.Done():
			return
		case <-tick.C:
		}
		r.mu.Lock()
		n := copy(in, r.queue)
		clear(in[n:])
		r.queue = r.queue[:copy(r.queue, r.queue[n:])]
		if n > 0 && len(r.queue) == 0 {
			r.userEnded = time.Now()
		}
		if r.userFed += n; n > 0 && r.userFed > r.userLead && r.userBegan.IsZero() {
			r.userBegan = time.Now()
		}
		r.mu.Unlock()
		state, err := r.cfg.Agent.Step(in, out)
		if err != nil {
			return
		}
		var energy float64
		for _, v := range out {
			energy += float64(v * v)
		}
		loud := state == speech.Speaking && math.Sqrt(energy/float64(len(out))) > 0.005
		r.mu.Lock()
		if r.state == speech.Speaking && state != speech.Speaking {
			r.quietAt, r.quietMark, r.resumedAt = time.Now(), len(r.spoke), time.Time{}
		}
		r.state = state
		if loud {
			now := time.Now()
			if !r.speaking {
				r.speaking = true
			}
			if r.firstSound.IsZero() {
				r.firstSound = now
			}
			if !r.quietAt.IsZero() && r.resumedAt.IsZero() {
				r.resumedAt = now
			}
			r.lastSound = now
			r.spoke = append(r.spoke, out...)
		} else if r.speaking && time.Since(r.lastSound) > 400*time.Millisecond {
			r.speaking = false
		}
		r.mu.Unlock()
	}
}

// clearly bounds the words of a user's line the harness's own ears may
// mishear; a line misheard more is said again, up to sayings times.
const (
	clearly = 0.2
	sayings = 4
)

// speak synthesizes text in the user's voice and queues it for the agent,
// then waits until it has been heard. The user speaks clearly: a synthesis
// that the harness's ears mishear is said again, and the clearest is
// spoken, so that a scenario tests the agent rather than one draw of the
// voice.
func (r *runner) speak(text string, lang speech.Language) error {
	opts := r.cfg.Speak
	if lang != speech.Unknown {
		opts.Language = lang
	}
	var line []float32
	for try, missed := 0, math.Inf(1); try < sayings && missed > clearly; try++ {
		pcm, err := speech.Synthesize(r.ctx, r.cfg.Voice, opts, text, nil)
		if err != nil {
			return err
		}
		mono, err := r.mono16k(pcm, r.cfg.Voice.SampleRate())
		if err != nil {
			return err
		}
		heard, err := r.hear16k(mono, lang)
		if err != nil {
			return err
		}
		if e := wordError(text, heard); e < missed {
			line, missed = mono, e
		}
		if missed > clearly && r.cfg.Log != nil {
			r.cfg.Log("the user's %q sounded like %q: said again", text, heard)
		}
	}
	lead := 0
	for lead < len(line) && math.Abs(float64(line[lead])) < 0.01 {
		lead++
	}
	r.mu.Lock()
	r.queue = append(r.queue, line...)
	r.spoke, r.firstSound = r.spoke[:0], time.Time{}
	r.userLead, r.userFed, r.userBegan = lead, 0, time.Time{}
	r.lastUser = text
	r.mu.Unlock()
	for {
		r.mu.Lock()
		left := len(r.queue)
		r.mu.Unlock()
		if left == 0 {
			return nil
		}
		select {
		case <-r.ctx.Done():
			return r.ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
	}
}

// wait lets d pass.
func (r *runner) wait(d time.Duration) error {
	select {
	case <-r.ctx.Done():
		return r.ctx.Err()
	case <-time.After(d):
		return nil
	}
}

// silent watches for d and reports the first sound, if any.
func (r *runner) silent(d time.Duration) (sounded bool, err error) {
	end := time.Now().Add(d)
	for time.Now().Before(end) {
		r.mu.Lock()
		s := r.speaking
		r.mu.Unlock()
		if s {
			return true, nil
		}
		if err := r.wait(20 * time.Millisecond); err != nil {
			return false, err
		}
	}
	return false, nil
}

// speaks waits up to d for the agent to speak, then for it to finish, and
// returns its audio and when it began.
func (r *runner) speaks(d time.Duration) (pcm []float32, after time.Duration, err error) {
	end := time.Now().Add(d)
	for {
		r.mu.Lock()
		s := r.speaking
		r.mu.Unlock()
		if s {
			break
		}
		if !time.Now().Before(end) {
			return nil, 0, nil
		}
		if err := r.wait(20 * time.Millisecond); err != nil {
			return nil, 0, err
		}
	}
	// Until the reply is over: the agent is back to listening (a model
	// may pause the voice mid-reply while it writes) and quiet.
	for {
		r.mu.Lock()
		quiet := !r.speaking && r.state == speech.Listening && time.Since(r.lastSound) > 600*time.Millisecond
		r.mu.Unlock()
		if quiet {
			break
		}
		if err := r.wait(20 * time.Millisecond); err != nil {
			return nil, 0, err
		}
	}
	r.mu.Lock()
	pcm = append([]float32(nil), r.spoke...)
	r.spoke = r.spoke[:0]
	if !r.firstSound.IsZero() && !r.userEnded.IsZero() {
		after = r.firstSound.Sub(r.userEnded)
	}
	r.firstSound = time.Time{}
	r.mu.Unlock()
	return pcm, after, nil
}

// hear transcribes the agent's audio.
func (r *runner) hear(pcm []float32, lang speech.Language) (string, error) {
	mono, err := r.mono16k(pcm, r.outRate)
	if err != nil {
		return "", err
	}
	return r.hear16k(mono, lang)
}

// mono16k resamples mono pcm at rate to 16 kHz.
func (r *runner) mono16k(pcm []float32, rate int) ([]float32, error) {
	n, err := speech.Samples16k(len(pcm), rate, 1)
	if err != nil {
		return nil, err
	}
	mono := make([]float32, n)
	n, err = r.resampler.Resample16kInto(pcm, rate, 1, mono)
	return mono[:n], err
}

// hear16k transcribes 16 kHz speech with the harness's ears.
func (r *runner) hear16k(mono []float32, lang speech.Language) (string, error) {
	opts := r.cfg.Listen
	opts.Partial, opts.Turn = nil, false
	if lang != speech.Unknown {
		opts.Language, opts.Languages = lang, 0
	}
	var t speech.Transcript
	if err := r.cfg.Ears.Transcribe(r.ctx, mono, opts, &t); err != nil {
		return "", err
	}
	return strings.TrimSpace(string(t.Text)), nil
}

// judge asks the judge whether heard says claim.
func (r *runner) judge(heard, claim string) (bool, string, error) {
	if r.cfg.Judge == nil {
		return false, "no judge", nil
	}
	s, err := r.cfg.Judge.NewSession("You check what a voice assistant said, from a transcript of its speech. " +
		"Answer with one word, yes or no.")
	if err != nil {
		return false, "", err
	}
	defer s.Close()
	if err := s.Add(chat.User, "Does the assistant say, or clearly mean, "+claim+"?\nThe assistant said: \""+heard+"\""); err != nil {
		return false, "", err
	}
	var b strings.Builder
	if err := s.Reply(r.ctx, chat.Options{MaxTokens: 4}, &b); err != nil {
		return false, "", err
	}
	verdict := strings.ToLower(strings.TrimSpace(b.String()))
	return strings.HasPrefix(verdict, "yes"), verdict, nil
}

// step performs one line.
func (r *runner) step(l line) (Step, error) {
	s := Step{Line: l.text}
	switch l.kind {
	case "user":
		r.stimulus()
		if err := r.speak(l.arg, l.lang); err != nil {
			return s, err
		}
		s.OK = true
	case "wait":
		r.stimulus()
		if err := r.wait(l.dur); err != nil {
			return s, err
		}
		s.OK = true
	case "note":
		r.stimulus()
		s.OK = r.note(l.arg, &s)
	case "chat":
		r.stimulus()
		name, text, _ := strings.Cut(l.arg, ":")
		s.OK = r.note(strings.TrimSpace(name)+" wrote in the call's chat: "+strings.TrimSpace(text), &s)
	case "join":
		r.stimulus()
		if r.speaker != nil {
			r.speaker(l.arg)
		}
		s.OK = r.note(l.arg+" joined the call.", &s)
	case "leave":
		r.stimulus()
		s.OK = r.note(l.arg+" left the call.", &s)
	case "say":
		r.stimulus()
		if err := r.cfg.Agent.Say(l.arg); err != nil {
			s.Reason = err.Error()
			return s, nil
		}
		s.OK = true
	case "silent":
		d := l.dur
		if d == 0 {
			d = r.cfg.Quiet
		}
		sounded, err := r.silent(d)
		if err != nil {
			return s, err
		}
		r.stimulus() // what follows is a new reply
		if sounded {
			pcm, _, _ := r.speaks(time.Second)
			s.Heard, _ = r.hear(pcm, l.lang)
			s.Reason = "the agent spoke"
			return s, nil
		}
		s.OK = true
	case "starts":
		d := l.dur
		if d == 0 {
			d = r.cfg.Answer
		}
		for end := time.Now().Add(d); ; {
			r.mu.Lock()
			started := r.speaking
			r.mu.Unlock()
			if started {
				break
			}
			if !time.Now().Before(end) {
				s.Reason = fmt.Sprintf("the agent said nothing in %v", d)
				return s, nil
			}
			if err := r.wait(20 * time.Millisecond); err != nil {
				return s, err
			}
		}
		r.fresh, r.lastPCM = true, nil
		s.OK = true
	case "stops":
		if r.cfg.Captions == nil {
			s.Reason = "no captions observer"
			return s, nil
		}
		r.mu.Lock()
		began := r.userBegan
		r.mu.Unlock()
		if began.IsZero() {
			s.Reason = "the user said nothing"
			return s, nil
		}
		if err := r.wait(time.Until(began.Add(l.dur))); err != nil {
			return s, err
		}
		at, ok := r.cfg.Captions.interruptedSince(began)
		if !ok || at.Sub(began) > l.dur {
			s.Reason = fmt.Sprintf("the agent went on through %v of the user's words", l.dur)
			return s, nil
		}
		s.Stopped, s.OK = at.Sub(began), true
		// What follows is the answer to the words that stopped the agent,
		// without what it said before them.
		r.mu.Lock()
		if !r.quietAt.Before(began) {
			cut := min(r.quietMark, len(r.spoke))
			r.spoke, r.firstSound = r.spoke[:copy(r.spoke, r.spoke[cut:])], r.resumedAt
		}
		r.mu.Unlock()
		r.fresh, r.lastPCM = true, nil
	case "goes on":
		r.mu.Lock()
		began := r.userBegan
		r.mu.Unlock()
		if r.cfg.Captions == nil || began.IsZero() {
			s.Reason = "no captions observer, or the user said nothing"
			return s, nil
		}
		pcm, _, err := r.speaks(r.cfg.Answer)
		if err != nil {
			return s, err
		}
		if _, stopped := r.cfg.Captions.interruptedSince(began); stopped {
			s.Heard, _ = r.hear(pcm, l.lang)
			s.Reason = "the user's words stopped the agent"
			return s, nil
		}
		r.fresh, r.lastPCM = true, nil
		s.OK = true
	case "speaks", "says", "repeats", "captions":
		if r.fresh || r.lastPCM == nil {
			d := l.dur
			if d == 0 {
				d = r.cfg.Answer
			}
			pcm, after, err := r.speaks(d)
			if err != nil {
				return s, err
			}
			if pcm == nil {
				s.Reason = fmt.Sprintf("the agent said nothing in %v", d)
				return s, nil
			}
			r.lastPCM, r.lastAfter, r.lastHeard, r.lastLang, r.fresh = pcm, after, "", speech.Unknown, false
		}
		if r.lastHeard == "" || r.lastLang != l.lang {
			var err error
			if r.lastHeard, err = r.hear(r.lastPCM, l.lang); err != nil {
				return s, err
			}
			r.lastLang = l.lang
		}
		s.Heard, s.After = r.lastHeard, r.lastAfter
		switch l.kind {
		case "speaks":
			s.OK = true
		case "says":
			ok, verdict, err := r.judge(s.Heard, l.arg)
			if err != nil {
				return s, err
			}
			s.OK = ok
			if !ok {
				s.Reason = "the judge said " + verdict
			}
		case "repeats":
			r.mu.Lock()
			user := r.lastUser
			r.mu.Unlock()
			s.OK = !strings.Contains(normalize(s.Heard), normalize(user))
			if !s.OK {
				s.Reason = "the agent repeated the user"
			}
		case "captions":
			if r.cfg.Captions == nil {
				s.Reason = "no captions observer"
				return s, nil
			}
			caps := r.cfg.Captions.since(r.stimulusAt)
			partial, final := false, false
			for _, c := range caps {
				if c.final {
					final = true
				} else if c.voiced > 0 && c.voiced < len(c.text) {
					partial = true
				}
			}
			s.OK = partial && final
			if !s.OK {
				s.Reason = fmt.Sprintf("%d captions, partial %v, final %v", len(caps), partial, final)
			}
		}
	default:
		return s, fmt.Errorf("unknown step %q", l.kind)
	}
	return s, nil
}

func (r *runner) note(text string, s *Step) bool {
	if err := r.cfg.Agent.Note(text); err != nil {
		s.Reason = err.Error()
		return false
	}
	return true
}

// wordError is the share of said's words that heard misses: their edit
// distance, compared normalized, over said's count.
func wordError(said, heard string) float64 {
	a, b := strings.Fields(normalize(said)), strings.Fields(normalize(heard))
	if len(a) == 0 {
		return 0
	}
	prev, cur := make([]int, len(b)+1), make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			sub := prev[j-1]
			if a[i-1] != b[j-1] {
				sub++
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, sub)
		}
		prev, cur = cur, prev
	}
	return float64(prev[len(b)]) / float64(len(a))
}

// normalize lowercases text and drops punctuation.
func normalize(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		switch {
		case r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r > 127:
			b.WriteRune(r)
		case r == ' ':
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

var errSyntax = errors.New("scenario: cannot read line")
