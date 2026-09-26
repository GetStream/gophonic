// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package duplex builds speech.Duplex agents from gophonic's lanes. A
// Cascade listens with a voice detector and a turn detector, understands
// with a transcriber, thinks with a chat session, and speaks with a
// synthesizer, all concurrently. It transcribes speech while it is spoken,
// keeping the conversation evaluated up to what has been said, and prepares
// its answer at the speaker's first pause, playing it the moment the turn
// detector agrees the turn is over: almost nothing is left to compute once
// the speaker is done. It drops the answer when the speaker goes on, stops
// within one frame when interrupted, and remembers only what was actually
// heard. The application sees no turns, only audio in and out:
//
//	agent, err := duplex.New(duplex.Config{Prompt: "You are Gopher."}, asr, turn, llm, tts)
//	for each 20 ms frame { state, err := agent.Step(ctx, in, out) }
//
// New finds the lanes it needs among the models it is given, by what they
// provide; any lane set in Config takes precedence, so a custom
// transcriber, detector, conversation, or voice composes in unchanged.
package duplex

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/gopus"
)

const (
	inRate    = speech.SampleRate // 16 kHz input, as the transcriber and turn detector take
	frameTime = 20 * time.Millisecond
	inFrame   = inRate / 50
	// Speech detection: SILK voice activity (0–255) that counts as speech,
	// and a level under which a frame is silence whatever the detector says
	// (-45 dBFS).
	voiced = 96
	quiet  = 0.0056
	// Turn taking.
	preroll = 300 * time.Millisecond  // audio kept before speech is detected: the detector fires late
	pause   = 100 * time.Millisecond  // silence that ends a turn the detector is sure of, when the words are unknown
	check   = 40 * time.Millisecond   // the turn is judged this often in a pause
	giveUp  = 1500 * time.Millisecond // silence that ends a turn whatever the detector says
	// The language model's log10 probability that the words end where the
	// speaker paused: above finishedWords they ask or say something whole;
	// below unfinishedWords they stop mid-thought, and the turn waits for
	// more, up to giveUpUnfinished.
	finishedWords    = -2.5
	unfinishedWords  = -7
	giveUpUnfinished = 3 * time.Second
	shortest         = 300 * time.Millisecond // speech an utterance needs
	resume           = 160 * time.Millisecond // speech that shows a speaker was not finished after all
	// Speech this soon into the agent's answer continues the speaker's
	// turn: the answer stops and is forgotten, and the whole utterance is
	// heard again.
	resumeWindow = 800 * time.Millisecond
	// Speech over the agent: after a short pause it is transcribed and
	// judged (an acknowledgement lets the agent go on; anything addressed
	// to it stops it), and talking over it this long stops it outright.
	overlapEnd  = 160 * time.Millisecond
	overlapPeek = 560 * time.Millisecond // talk over the agent this long is judged while it goes on
	overlapLong = 1500 * time.Millisecond
	// Text speed, for estimating how much of an interrupted reply was heard.
	charsPerSecond = 14
	// Transcription while speaking: the speech an utterance needs before it
	// is first transcribed, and the new speech that starts another pass.
	scribeFirst = inRate / 2
	longest     = 60 * inRate // samples of an utterance heard; more is dropped
	scribeStep  = inRate * 160 / 1000
)

// Config shapes a Cascade. Every field is optional.
type Config struct {
	// Prompt is the system prompt of a conversation New starts.
	Prompt string
	// The lanes. New opens those left nil from its models; the Cascade
	// owns all of them and closes them in Close.
	Transcriber  speech.Transcriber
	TurnDetector speech.TurnDetector
	Session      chat.Session // the conversation, its system prompt added
	Synthesizer  speech.Synthesizer
	// Listen configures transcription, such as the language spoken.
	Listen speech.Options
	// Voice selects the synthesizer's voice and language.
	Voice speech.SpeakOptions
	// Reply shapes the language model's answers.
	Reply chat.Options
	// Interruptions judges speech over the agent's voice: given the text
	// "Assistant: <what it was saying>\nUser: <what was said over it>", its
	// first label means go on and its second means stop. Nil uses a zero-
	// shot classifier from New's models when one provides it, and
	// otherwise a lexicon of acknowledgements and stop words.
	Interruptions speech.TextClassifier
	// Heard, when not nil, sees each transcribed utterance, and what has
	// been said so far while it is spoken, before it joins the
	// conversation: it returns the message to add, such as the text with
	// its speaker's name, and whether the agent should answer it, as in a
	// meeting where people also talk to each other. Utterances left
	// unanswered still join the conversation, so the agent knows them when
	// asked. It runs on a worker goroutine. Nil adds the text and answers
	// everything.
	Heard func(text string) (message string, answer bool)
	// OnText, when not nil, receives what is said, from a worker
	// goroutine: each transcribed utterance (chat.User), and each reply
	// (chat.Assistant) as it grows, then once more, final, when it is
	// finished or interrupted. Captions show the growing text; a record
	// keeps the final one.
	OnText func(role chat.Role, text string, final bool)
	// OnError, when not nil, receives errors of the workers.
	OnError func(error)
	// OnStage, when not nil, receives each reply's progress from the
	// speaker's pause: "transcribed", "judged" (whether the words are
	// finished), "first text", "first audio", and "turn" when the turn is
	// found over, followed by its evidence; the answer plays from the later
	// of the last two.
	OnStage func(stage string, elapsed time.Duration)
}

// Cascade is a speech.Duplex made of lanes; see the package comment.
type Cascade struct {
	cfg              Config
	outRate, outSize int

	// Step's side: input frames to the listener, output from playback.
	in      *ring[float32] // input samples for the listener
	inReady chan struct{}
	play    *ring[float32] // synthesized samples waiting to be played
	played  atomic.Int64   // samples of the current reply played
	// A reply prepared at a pause is held, unplayed, until the turn ends.
	held     atomic.Bool
	resumed  atomic.Bool   // the current reply was cut by its speaker going on
	seq      atomic.Uint32 // the last job's id
	released atomic.Uint32 // the id of the job whose turn ended
	release  chan struct{} // signals released

	mu    sync.Mutex
	state speech.DuplexState

	asr    sync.Mutex // the transcriber serves the scribe, listener, and responder
	saying string     // what the agent is saying; guarded by mu
	probs  []float32  // Interruptions' output

	// The utterance being heard, written by the listener and transcribed
	// by the scribe as it grows; partial is the scribe's latest transcript
	// of it.
	utt       utterance
	uttReady  chan struct{}
	pauses    atomic.Uint32 // speaker pauses so far: each aborts the scribe's pass
	pass      passContext
	partialMu sync.Mutex
	partial   speech.Transcript
	partialOf uint32 // the utterance generation partial transcribes
	partialN  int    // its samples
	prefill   chan struct{}
	// The language model's judgment of the pending answer's words: the
	// log10 probability that they end the speaker's message.
	wordsMu  sync.Mutex
	wordsJob uint32
	words    float32

	jobs   chan job
	notes  []note             // messages to add between replies; guarded by mu
	noted  chan struct{}      // signals notes
	cancel context.CancelFunc // cancels the current reply; guarded by mu
	busy   atomic.Bool        // a reply is being prepared or spoken
	stop   chan struct{}
	wg     sync.WaitGroup
	closed bool
}

// note is a message added without a reply.
type note struct {
	role chat.Role
	text string
}

// job is one reply to produce: an utterance to transcribe and answer, or
// text to say. A held job plays only once its turn is released.
type job struct {
	audio []float32
	say   string
	utt   uint32    // the utterance generation of audio
	id    uint32    // set by dispatch
	held  bool      // prepared at a pause, before the turn is known to be over
	at    time.Time // the speaker's pause
}

// utterance is the audio of the speech being heard. Only the listener
// writes it; it reads without the lock, the scribe with it.
type utterance struct {
	mu  sync.Mutex
	pcm []float32
	gen uint32 // counts utterances
}

func (u *utterance) append(v []float32) {
	u.mu.Lock()
	if len(u.pcm)+len(v) <= cap(u.pcm) {
		u.pcm = append(u.pcm, v...)
	}
	u.mu.Unlock()
}

func (u *utterance) reset() {
	u.mu.Lock()
	u.pcm = u.pcm[:0]
	u.gen++
	u.mu.Unlock()
}

// passContext is the context of the scribe's passes: its Err reports a
// pass aborted once the speaker pauses, so the transcription of the answer
// need not wait for it. It allocates nothing per pass.
type passContext struct {
	context.Context
	c      *Cascade
	pauses uint32 // the pauses when the pass began
}

func (p *passContext) Err() error {
	if p.c.pauses.Load() != p.pauses {
		return context.Canceled
	}
	return p.Context.Err()
}

var _ speech.Duplex = (*Cascade)(nil)

// New starts a Cascade from cfg, opening each lane cfg leaves nil from the
// first of models that provides it: a speech.Transcriber, a
// speech.TurnDetector, a chat.Generator for the conversation, and a
// speech.Synthesizer.
func New(cfg Config, models ...*gophonic.Model) (_ *Cascade, err error) {
	var opened []interface{ Close() error }
	defer func() {
		if err != nil {
			for _, l := range opened {
				l.Close()
			}
		}
	}()
	if err := open(&cfg.Transcriber, models, &opened); err != nil {
		return nil, err
	}
	if err := open(&cfg.TurnDetector, models, &opened); err != nil {
		return nil, err
	}
	if err := open(&cfg.Synthesizer, models, &opened); err != nil {
		return nil, err
	}
	if cfg.Interruptions == nil {
		var z speech.ZeroShot
		if open(&z, models, &opened) == nil {
			if cfg.Interruptions, err = z.Classifier(interruptQuestion, interruptLabels); err != nil {
				return nil, err
			}
			opened = append(opened, cfg.Interruptions)
		}
	}
	if cfg.Session == nil {
		var g chat.Generator
		if err := open(&g, models, &opened); err != nil {
			return nil, err
		}
		if cfg.Session, err = g.NewSession(cfg.Prompt); err != nil {
			return nil, err
		}
		opened = append(opened, cfg.Session)
	}
	out := cfg.Synthesizer.SampleRate()
	c := &Cascade{cfg: cfg, outRate: out, outSize: out / 50,
		in: newRing[float32](2 * inRate), inReady: make(chan struct{}, 1),
		play: newRing[float32](120 * out), release: make(chan struct{}, 1),
		utt: utterance{pcm: make([]float32, 0, longest)}, uttReady: make(chan struct{}, 1), prefill: make(chan struct{}, 1),
		jobs: make(chan job, 4), noted: make(chan struct{}, 1), stop: make(chan struct{}), probs: make([]float32, 2)}
	c.pass = passContext{Context: context.Background(), c: c}
	vad, err := gopus.NewVAD(inRate)
	if err != nil {
		return nil, err
	}
	if err := c.warm(); err != nil {
		return nil, err
	}
	c.wg.Add(3)
	go c.listen(vad)
	go c.scribe()
	go c.respond()
	return c, nil
}

// open sets *lane, when nil, to a lane of the first model that provides
// its type.
func open[T interface{ Close() error }](lane *T, models []*gophonic.Model, opened *[]interface{ Close() error }) error {
	if any(*lane) != nil {
		return nil
	}
	for _, m := range models {
		if gophonic.Supports[T](m) {
			l, err := gophonic.Lane[T](m)
			if err != nil {
				return err
			}
			*lane = l
			*opened = append(*opened, l)
			return nil
		}
	}
	return fmt.Errorf("duplex: no model provides %v: %w", reflect.TypeFor[T](), speech.ErrUnsupported)
}

// warm runs the transcriber and the synthesizer once, so that their first
// use in a conversation is as fast as every later one.
func (c *Cascade) warm() error {
	var t speech.Transcript
	if err := c.cfg.Transcriber.Transcribe(context.Background(), make([]float32, inRate), c.cfg.Listen, &t); err != nil {
		return err
	}
	said := false
	return c.cfg.Synthesizer.Speak(context.Background(), c.cfg.Voice, func() ([]byte, error) {
		if said {
			return nil, io.EOF
		}
		said = true
		return []byte("Hi."), nil
	}, func([]float32) error { return nil })
}

// Rates: 16 kHz in, the synthesizer's rate out.
func (c *Cascade) Rates() (in, out int) { return inRate, c.outRate }

// Frame is 20 ms.
func (c *Cascade) Frame() (in, out int) { return inFrame, c.outSize }

// Step hands in to the listener and writes the next 20 ms of speech.
func (c *Cascade) Step(ctx context.Context, in, out []float32) (speech.DuplexState, error) {
	if len(in) != inFrame || len(out) != c.outSize {
		return speech.Listening, errors.New("duplex: Step takes one frame in and out")
	}
	c.in.write(in)
	select {
	case c.inReady <- struct{}{}:
	default:
	}
	n := 0
	if !c.held.Load() {
		n = c.play.read(out)
	}
	clear(out[n:])
	c.played.Add(int64(n))
	c.mu.Lock()
	defer c.mu.Unlock()
	switch {
	case n > 0:
		c.state = speech.Speaking
	case c.busy.Load():
		c.state = speech.Thinking
	default:
		c.state = speech.Listening
	}
	return c.state, ctx.Err()
}

// Buffered reports how much of the agent's speech is synthesized and
// waiting to be played.
func (c *Cascade) Buffered() time.Duration {
	return time.Duration(c.play.len()) * time.Second / time.Duration(c.outRate)
}

// Say speaks text as soon as the current reply ends.
func (c *Cascade) Say(text string) error {
	select {
	case c.jobs <- job{say: text}:
		return nil
	default:
		return errors.New("duplex: too many replies queued")
	}
}

// Add adds a message to the conversation without answering it, such as a
// participant's chat message, so that the agent knows it when asked. It
// joins the conversation between replies.
func (c *Cascade) Add(role chat.Role, text string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return chat.ErrClosed
	}
	c.notes = append(c.notes, note{role, text})
	select {
	case c.noted <- struct{}{}:
	default:
	}
	return nil
}

// addNotes adds the messages waiting to the conversation.
func (c *Cascade) addNotes() {
	c.mu.Lock()
	notes := c.notes
	c.notes = nil
	c.mu.Unlock()
	for _, n := range notes {
		if err := c.cfg.Session.Add(n.role, n.text); err != nil {
			c.fail(err)
		}
	}
}

// Close stops the workers and closes the lanes.
func (c *Cascade) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	if c.cancel != nil {
		c.cancel()
	}
	c.mu.Unlock()
	close(c.stop)
	c.wg.Wait()
	return errors.Join(c.cfg.Transcriber.Close(), c.cfg.TurnDetector.Close(), c.cfg.Session.Close(), c.cfg.Synthesizer.Close())
}

// interrupt stops the current reply: its audio stops at once and the
// responder records what was heard.
func (c *Cascade) interrupt() {
	c.mu.Lock()
	if c.cancel != nil {
		c.cancel()
		c.cancel = nil
	}
	c.mu.Unlock()
	c.play.reset()
}

// audible reports whether the agent's voice is playing, or about to.
func (c *Cascade) audible() bool {
	return c.busy.Load() && !c.held.Load() && (c.played.Load() > 0 || c.play.len() > 0)
}

// listen follows the room's audio frame by frame: voice activity marks
// speech; at a pause the answer is prepared and the turn detector asked
// whether the speaker is done, which releases it; and speech over the
// agent either lets it go on (an acknowledgement) or interrupts it.
func (c *Cascade) listen(vad *gopus.VAD) {
	defer c.wg.Done()
	var (
		frame    = make([]float32, inFrame)
		pcm16    = make([]int16, inFrame)
		keep     = int(preroll/frameTime) * inFrame
		early    = make([]float32, 0, keep)
		talked   time.Duration
		silence  time.Duration
		overlap  time.Duration // continuous speech so far
		paused   time.Time     // when the speaker last paused
		heardAt  time.Time     // when the agent's voice became audible
		pending  uint32        // the answer prepared at the pause, held until the turn ends
		answered bool          // the turn ended and its answer was released
		stopped  bool          // speech over the agent stopped it
	)
	reset := func() {
		c.utt.reset()
		talked, silence, pending, answered = 0, 0, 0, false
	}
	for {
		select {
		case <-c.stop:
			return
		case <-c.inReady:
		}
		for c.in.len() >= inFrame {
			c.in.read(frame)
			var energy float32
			for i, s := range frame {
				pcm16[i] = int16(max(-1, min(1, s)) * 32767)
				energy += s * s
			}
			activity, _ := vad.AnalyzeInt16(pcm16)
			speaking := activity >= voiced && energy >= quiet*quiet*inFrame
			if speaking {
				overlap += frameTime
			} else {
				overlap = 0
			}
			audible := c.audible()
			switch {
			case !audible:
				heardAt = time.Time{}
			case heardAt.IsZero():
				heardAt = time.Now()
			}
			fresh := audible && time.Since(heardAt) < resumeWindow
			switch {
			case !audible:
				stopped = false
			case answered && !fresh:
				// The agent's answer is under way: the utterance is done.
				reset()
			}
			if answered && !c.busy.Load() {
				reset() // answered without a sound, or not to be answered
			}
			if speaking && pending != 0 {
				// The speaker goes on: the answer prepared at the pause is
				// stale. Drop it and hear the whole utterance.
				c.interrupt()
				pending = 0
			}
			if answered && c.busy.Load() && (!audible || fresh) && overlap >= resume {
				// Speech before the answer is heard, or as it begins: the
				// speaker was not finished after all.
				c.resumed.Store(true)
				c.interrupt()
				answered = false
			}
			if len(c.utt.pcm) == 0 && !speaking {
				early = append(early, frame...)
				if over := len(early) - keep; over > 0 {
					early = early[:copy(early, early[over:])]
				}
				continue
			}
			if len(c.utt.pcm) == 0 {
				c.utt.append(early)
				early = early[:0]
			}
			c.utt.append(frame)
			if speaking {
				talked, silence = talked+frameTime, 0
				select {
				case c.uttReady <- struct{}{}:
				default:
				}
			} else {
				silence += frameTime
			}
			utterance := c.utt.pcm
			if audible && !stopped && !answered {
				// Speech over the agent is judged once it pauses, or stops
				// the agent if it goes on; turns wait until the agent is
				// silent.
				switch {
				case talked >= overlapLong:
					c.interrupt()
					stopped = true
				case speaking && talked == overlapPeek:
					// Long enough to be more than an acknowledgement:
					// judge what has been said so far.
					if c.interrupts(utterance) {
						c.interrupt()
						stopped = true
					}
				case talked > 0 && silence == overlapEnd:
					if c.interrupts(utterance) {
						c.interrupt()
						stopped = true
					} else {
						reset()
					}
				}
				continue
			}
			if speaking || answered {
				continue
			}
			if pending == 0 && talked >= shortest {
				// The speaker paused: prepare the answer now, and play it
				// if the turn turns out to be over.
				paused = time.Now()
				c.pauses.Add(1)
				pending = c.dispatch(job{audio: append([]float32(nil), utterance...), utt: c.utt.gen, held: true, at: paused})
			}
			if silence%check != 0 && silence < giveUp {
				continue
			}
			p, err := c.cfg.TurnDetector.PredictInto(utterance, inRate, 1)
			if err != nil {
				c.fail(err)
				continue
			}
			words, known := c.wordsOf(pending)
			if !turnOver(silence, p, words, known) {
				continue
			}
			if c.cfg.OnStage != nil && !paused.IsZero() {
				detail := fmt.Sprintf("turn (sound %.2f)", p.Probability)
				if known {
					detail = fmt.Sprintf("turn (sound %.2f, words %.1f)", p.Probability, words)
				}
				c.cfg.OnStage(detail, time.Since(paused))
			}
			if talked < shortest {
				reset() // a cough or a click
				continue
			}
			if pending == 0 {
				paused = time.Now()
				pending = c.dispatch(job{audio: append([]float32(nil), utterance...), utt: c.utt.gen, at: paused})
			}
			c.releaseTurn(pending)
			pending, answered = 0, true
		}
	}
}

// turnOver decides whether a pause of silence ends the speaker's turn, from
// the turn detector's prediction and, when known, the language model's
// log10 probability that the words end there: finished words need little
// from the detector, unfinished ones wait for it to be sure.
func turnOver(silence time.Duration, p speech.Prediction, words float32, known bool) bool {
	switch {
	case !known:
		return silence >= pause && p.Complete || silence >= giveUp
	case words >= finishedWords:
		// Whole words; the detector can hold the turn open a little
		// while, as a speaker pausing mid-sentence sounds unfinished.
		return p.Probability >= 0.2 || silence >= 4*check
	case words > unfinishedWords:
		return silence >= 2*check && p.Complete || silence >= giveUp
	default:
		return silence >= 500*time.Millisecond && p.Probability >= 0.9 || silence >= giveUpUnfinished
	}
}

// wordsOf returns the language model's judgment of job id's words, once
// the responder has it.
func (c *Cascade) wordsOf(id uint32) (float32, bool) {
	c.wordsMu.Lock()
	defer c.wordsMu.Unlock()
	return c.words, id != 0 && c.wordsJob == id
}

// scribe transcribes the utterance while it is spoken, each pass
// continuing the last, so that at a pause only its end is left to
// transcribe, and has the responder evaluate the conversation up to what
// has been said.
func (c *Cascade) scribe() {
	defer c.wg.Done()
	var (
		pcm       = make([]float32, 0, longest)
		prev, cur speech.Transcript
		gen       uint32
		done      int // samples transcribed
		// New speech that starts a pass: at least twice the last pass's
		// time, so that long speech leaves the GPU half free.
		step = scribeStep
	)
	for {
		select {
		case <-c.stop:
			return
		case <-c.uttReady:
		}
		c.utt.mu.Lock()
		if c.utt.gen != gen {
			gen, done = c.utt.gen, 0
			prev.Reset()
		}
		n := len(c.utt.pcm)
		if n < scribeFirst || n-done < step {
			c.utt.mu.Unlock()
			continue
		}
		pcm = append(pcm[:0], c.utt.pcm...)
		c.utt.mu.Unlock()
		opts := c.cfg.Listen
		if len(prev.Text) > 0 {
			opts.Partial = &prev
		}
		c.pass.pauses = c.pauses.Load()
		began := time.Now()
		c.asr.Lock()
		err := c.cfg.Transcriber.Transcribe(&c.pass, pcm, opts, &cur)
		c.asr.Unlock()
		step = max(scribeStep, int(2*time.Since(began)*inRate/time.Second))
		if err != nil {
			if c.pass.Err() == nil {
				c.fail(err)
			}
			continue
		}
		done = n
		prev, cur = cur, prev
		c.partialMu.Lock()
		changed := c.partialOf != gen || string(c.partial.Text) != string(prev.Text)
		c.partial.Text = append(c.partial.Text[:0], prev.Text...)
		c.partial.Language, c.partialOf, c.partialN = prev.Language, gen, n
		c.partialMu.Unlock()
		if changed {
			select {
			case c.prefill <- struct{}{}:
			default:
			}
		}
	}
}

// partialFor copies the scribe's latest transcript of utterance gen, if
// any, to dst.
func (c *Cascade) partialFor(gen uint32, dst *speech.Transcript) bool {
	c.partialMu.Lock()
	defer c.partialMu.Unlock()
	if c.partialOf != gen || len(c.partial.Text) == 0 {
		return false
	}
	dst.Text = append(dst.Text[:0], c.partial.Text...)
	dst.Language = c.partial.Language
	return true
}

// heard applies Config.Heard to text.
func (c *Cascade) heard(text string) (string, bool) {
	if c.cfg.Heard == nil {
		return text, true
	}
	return c.cfg.Heard(text)
}

// prefillPartial judges what the speaker has said so far, as the answer
// will: that evaluates the conversation up to it, and when the utterance
// ends as it was heard, the answer's judgment is already made.
func (c *Cascade) prefillPartial() {
	c.partialMu.Lock()
	text := strings.TrimSpace(string(c.partial.Text))
	c.partialMu.Unlock()
	if unclosed(text) == "" {
		return
	}
	message, _ := c.heard(text)
	if _, err := c.cfg.Session.Finished(context.Background(), chat.User, message); err != nil {
		c.fail(err)
	}
}

// releaseTurn lets the answer of job id play: its speaker is done.
func (c *Cascade) releaseTurn(id uint32) {
	if id == 0 {
		return
	}
	c.released.Store(id)
	c.held.Store(false)
	select {
	case c.release <- struct{}{}:
	default:
	}
}

// awaitTurn waits for job id's turn to end, reporting false if the job is
// dropped first.
func (c *Cascade) awaitTurn(ctx context.Context, id uint32) bool {
	for c.released.Load() != id {
		select {
		case <-ctx.Done():
			return false
		case <-c.release:
		}
	}
	return true
}

// The zero-shot question that judges speech over the agent.
const interruptQuestion = `A voice assistant was speaking when the user said something over it.
Does the user want the assistant to stop and listen, judging from what each said?`

var interruptLabels = []string{
	"No: it is an acknowledgement, agreement, a reaction, laughter, or noise; the assistant should go on",
	"Yes: the user objects, corrects, asks, or wants to say something; the assistant should stop",
}

// interrupts reports whether speech heard over the agent means it should
// stop: acknowledgements and noise do not; stop words do; anything else is
// judged in the context of what the agent is saying.
func (c *Cascade) interrupts(audio []float32) bool {
	var t, partial speech.Transcript
	opts := c.cfg.Listen
	if c.partialFor(c.utt.gen, &partial) {
		opts.Partial = &partial
	}
	c.asr.Lock()
	err := c.cfg.Transcriber.Transcribe(context.Background(), audio, opts, &t)
	c.asr.Unlock()
	if err != nil {
		c.fail(err)
		return true
	}
	heard := normalize(string(t.Text))
	stop := c.judge(heard, string(t.Text))
	if c.cfg.OnStage != nil {
		c.cfg.OnStage(fmt.Sprintf("heard %q over the agent: stop %v", heard, stop), 0)
	}
	return stop
}

// judge decides whether heard, normalized from text, stops the agent.
func (c *Cascade) judge(heard, text string) bool {
	switch {
	case heard == "" || acknowledgements[heard]:
		return false
	case stopWords.MatchString(heard):
		return true
	case c.cfg.Interruptions == nil:
		return len(strings.Fields(heard)) > 2
	}
	c.mu.Lock()
	saying := c.saying
	c.mu.Unlock()
	if len(saying) > 300 {
		saying = "…" + saying[len(saying)-300:]
	}
	input := "Assistant: " + saying + "\nUser: " + strings.TrimSpace(text)
	if err := c.cfg.Interruptions.ClassifyInto(context.Background(), input, c.probs); err != nil {
		c.fail(err)
		return true
	}
	return c.probs[1] > c.probs[0]
}

// speakable drops what a voice cannot say and captions need not show:
// emoji and other pictographs, and markdown emphasis and headings.
func speakable(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r == '*' || r == '_' || r == '#' || r == '`' || r == '~':
			return -1
		case r == 0x200d || r >= 0xfe00 && r <= 0xfe0f: // joiners and variation selectors
			return -1
		case unicode.Is(unicode.So, r) || unicode.Is(unicode.Sk, r) || r >= 0x1f000:
			return -1
		}
		return r
	}, s)
}

// unclosed drops closing punctuation: a transcript of it alone is empty.
func unclosed(text string) string {
	return strings.TrimRightFunc(text, func(r rune) bool { return unicode.IsPunct(r) || unicode.IsSpace(r) })
}

// normalize lowercases text and drops punctuation.
func normalize(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, s)
	return strings.Join(strings.Fields(s), " ")
}

// acknowledgements are things listeners say to keep a speaker going.
var acknowledgements = map[string]bool{}

func init() {
	for _, a := range strings.Split(`yeah|yes|yep|yup|ya|mhm|mm|mmm|mm hmm|mmhmm|uh huh|uhhuh|hmm|hm|right|okay|ok|sure|i see|got it|cool|nice|great|wow|oh|ah|aha|haha|ha|true|exactly|totally|indeed|interesting|go on|really|no way|oh wow|oh nice|sim|é|pois|claro|pois é|tá|ok ok|sí|vale|claro que sí|ja|genau|oui|d'accord|嗯|对|好|是的`, "|") {
		acknowledgements[a] = true
	}
}

// stopWords always stop the agent.
var stopWords = regexp.MustCompile(`\b(stop|wait|hold on|hang on|shut up|enough|pause|be quiet|excuse me|sorry|para|espera|arrête|halt)\b`)

// dispatch replaces any reply in preparation with j and returns its id, or
// zero if it was dropped.
func (c *Cascade) dispatch(j job) uint32 {
	c.interrupt()
	j.id = c.seq.Add(1)
	c.held.Store(j.held)
	c.busy.Store(true)
	select {
	case c.jobs <- j:
		return j.id
	default:
		c.fail(errors.New("duplex: responder is behind; utterance dropped"))
		return 0
	}
}

// respond produces replies one at a time: it transcribes, asks the chat
// session, and streams the answer's text into the synthesizer, whose
// audio queues for Step.
func (c *Cascade) respond() {
	defer c.wg.Done()
	var t, partial speech.Transcript
	// The language model runs only a couple of pieces ahead of the voice:
	// the voice synthesizes faster than real time, so the two alternate on
	// the GPU instead of queueing behind each other. Until the first audio
	// it writes only what the voice asks for, since the listener waits for
	// that frame and not for the words after it.
	const ahead = 2
	pieces := make(chan string, ahead)
	for {
		var j job
		select {
		case <-c.stop:
			return
		case <-c.noted:
			c.addNotes()
			continue
		case <-c.prefill:
			c.prefillPartial()
			continue
		case j = <-c.jobs:
		}
		// A newer job waiting supersedes this one.
		for len(c.jobs) > 0 {
			j = <-c.jobs
		}
		c.addNotes()
		c.resumed.Store(false)
		if !j.held {
			c.held.Store(false)
		}
		ctx, cancel := context.WithCancel(context.Background())
		c.mu.Lock()
		c.cancel = cancel
		c.mu.Unlock()
		if j.at.IsZero() {
			j.at = time.Now()
		}
		trace := func(stage string) {
			if c.cfg.OnStage != nil {
				c.cfg.OnStage(stage, time.Since(j.at))
			}
		}
		c.played.Store(0)
		mark := c.cfg.Session.Checkpoint()
		text := j.say
		if j.audio != nil {
			opts := c.cfg.Listen
			if c.partialFor(j.utt, &partial) {
				opts.Partial = &partial
			}
			c.asr.Lock()
			err := c.cfg.Transcriber.Transcribe(ctx, j.audio, opts, &t)
			c.asr.Unlock()
			if err != nil {
				if ctx.Err() != nil {
					err = nil // superseded
				}
				c.finish(cancel, err)
				continue
			}
			trace("transcribed")
			text = strings.TrimSpace(string(t.Text))
			if unclosed(text) == "" {
				c.finish(cancel, nil)
				continue
			}
			message, answer := c.heard(text)
			// Whether the words are finished helps judge the turn; their
			// evaluation is the reply's first step anyway.
			if p, err := c.cfg.Session.Finished(ctx, chat.User, message); err == nil {
				c.wordsMu.Lock()
				c.wordsJob, c.words = j.id, float32(math.Log10(max(float64(p), 1e-30)))
				c.wordsMu.Unlock()
				trace("judged")
			} else if ctx.Err() == nil {
				c.fail(err)
			}
			if err := c.cfg.Session.Add(chat.User, message); err != nil {
				c.finish(cancel, err)
				continue
			}
			if !answer {
				// Not meant for the agent: it joins the conversation, and
				// shows, once the turn is over.
				if c.awaitTurn(ctx, j.id) {
					if c.cfg.OnText != nil {
						c.cfg.OnText(chat.User, text, true)
					}
					if err := c.cfg.Session.Prefill(ctx); err != nil && ctx.Err() == nil {
						c.fail(err)
					}
				} else {
					c.cfg.Session.Restore(mark)
				}
				c.finish(cancel, nil)
				continue
			}
		}
		// The synthesizer reads text as the model writes it.
		var reply strings.Builder
		// The utterance is shown once its answer is certain to be heard,
		// not while it might still be superseded or continued, and the
		// answer's captions follow it.
		userShown := false
		showUser := func() {
			if !userShown && j.audio != nil && c.cfg.OnText != nil &&
				c.played.Load() >= int64(resumeWindow)*int64(c.outRate)/int64(time.Second) {
				userShown = true
				c.cfg.OnText(chat.User, text, true)
				c.cfg.OnText(chat.Assistant, speakable(reply.String()), false)
			}
		}
		var (
			asked  atomic.Int32             // pieces the synthesizer has asked for
			wanted = make(chan struct{}, 1) // signals asked
			sound  = make(chan struct{})    // closed at the first audio, or when the voice ends
			once   sync.Once
			sent   int32 // pieces written
		)
		sounded := func() { once.Do(func() { close(sound) }) }
		done := make(chan error, 1)
		go func() {
			defer sounded()
			done <- c.cfg.Synthesizer.Speak(ctx, c.cfg.Voice, func() ([]byte, error) {
				asked.Add(1)
				select {
				case wanted <- struct{}{}:
				default:
				}
				select {
				case p, ok := <-pieces:
					if !ok {
						return nil, io.EOF
					}
					return []byte(p), nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}, func(pcm []float32) error {
				if c.play.len() == 0 && c.played.Load() == 0 {
					trace("first audio")
				}
				c.play.write(pcm)
				sounded()
				return ctx.Err()
			})
		}()
		var err error
		if j.audio != nil {
			err = c.cfg.Session.Reply(ctx, c.cfg.Reply, func(p []byte) error {
				if reply.Len() == 0 {
					trace("first text")
				}
				reply.Write(p)
				spoken := speakable(string(p))
				c.mu.Lock()
				c.saying = reply.String()
				c.mu.Unlock()
				showUser()
				if userShown && strings.ContainsAny(string(p), " .,!?;:") {
					c.cfg.OnText(chat.Assistant, speakable(reply.String()), false)
				}
				if spoken == "" {
					return nil
				}
				select {
				case pieces <- spoken:
					sent++
				case <-ctx.Done():
					return ctx.Err()
				}
				// Before the first audio, wait until it sounds or the voice
				// needs more text.
				for asked.Load() <= sent {
					select {
					case <-sound:
						return nil
					case <-wanted:
					case <-ctx.Done():
						return ctx.Err()
					}
				}
				return nil
			})
		} else {
			reply.WriteString(text)
			pieces <- text
			c.cfg.Session.Add(chat.Assistant, text)
		}
		close(pieces)
		speakErr := <-done
		pieces = make(chan string, ahead)
		interrupted := c.waitPlayback(ctx, showUser)
		if c.cfg.OnText != nil && j.audio != nil && !userShown && text != "" && c.played.Load() > 0 && !c.resumed.Load() {
			// A reply heard only after the model finished writing it.
			c.cfg.OnText(chat.User, text, true)
		}
		said := speakable(reply.String())
		switch played := c.played.Load(); {
		case interrupted && played == 0:
			// Superseded before a sound: the conversation never had it.
			c.cfg.Session.Restore(mark)
			said, text = "", ""
		case interrupted && c.resumed.Load():
			// Cut as it began by the speaker going on: the utterance is
			// heard again whole, so the conversation and the captions
			// forget both.
			c.cfg.Session.Restore(mark)
			said = ""
		case interrupted:
			heard := min(len(reply.String()), int(time.Duration(played)*time.Second/time.Duration(c.outRate)*charsPerSecond/time.Second))
			if j.audio != nil {
				c.cfg.Session.Truncate(heard)
			}
			said = cut(said, heard) + "…"
		}
		if c.cfg.OnText != nil && said != "" {
			c.cfg.OnText(chat.Assistant, said, true)
		}
		if interrupted {
			err, speakErr = nil, nil
		}
		c.finish(cancel, errors.Join(err, speakErr))
	}
}

// waitPlayback waits for the reply to be heard, or cut, and reports whether
// it was interrupted. showUser publishes captions as playback progresses.
func (c *Cascade) waitPlayback(ctx context.Context, showUser func()) bool {
	interrupted := ctx.Err() != nil
	for !interrupted && c.play.len() > 0 {
		select {
		case <-ctx.Done():
			interrupted = true
		case <-time.After(frameTime):
			showUser()
		}
	}
	// interrupt cancels the context before clearing playback. The timer
	// branch can win that select, then an empty queue ends the loop without
	// ever selecting ctx.Done. Observe cancellation again at that boundary.
	return interrupted || ctx.Err() != nil
}

// finish ends the current reply.
func (c *Cascade) finish(cancel context.CancelFunc, err error) {
	cancel()
	c.mu.Lock()
	c.cancel = nil
	c.mu.Unlock()
	if len(c.jobs) == 0 {
		c.busy.Store(false)
	}
	if err != nil {
		c.fail(err)
	}
}

func (c *Cascade) fail(err error) {
	if c.cfg.OnError != nil {
		c.cfg.OnError(err)
	}
}

// cut returns the first n bytes of s, backed off to a UTF-8 boundary.
func cut(s string, n int) string {
	if n >= len(s) {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// ring is a bounded FIFO of samples, safe for one writer and one reader.
type ring[T any] struct {
	mu   sync.Mutex
	buf  []T
	head int // next to read
	n    int
}

func newRing[T any](capacity int) *ring[T] { return &ring[T]{buf: make([]T, capacity)} }

// write appends v, dropping the oldest samples when full.
func (r *ring[T]) write(v []T) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for len(v) > 0 {
		if r.n == len(r.buf) {
			r.head = (r.head + 1) % len(r.buf)
			r.n--
		}
		tail := (r.head + r.n) % len(r.buf)
		k := copy(r.buf[tail:min(len(r.buf), tail+len(r.buf)-r.n)], v)
		r.n += k
		v = v[k:]
	}
}

// read fills dst from the front and reports how many it read.
func (r *ring[T]) read(dst []T) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for n < len(dst) && r.n > 0 {
		k := copy(dst[n:], r.buf[r.head:min(len(r.buf), r.head+r.n)])
		r.head = (r.head + k) % len(r.buf)
		r.n -= k
		n += k
	}
	return n
}

func (r *ring[T]) len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.n
}

func (r *ring[T]) reset() {
	r.mu.Lock()
	r.head, r.n = 0, 0
	r.mu.Unlock()
}
