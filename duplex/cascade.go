// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package duplex builds speech.Duplex agents from gophonic's lanes. A
// Cascade listens with a voice detector and a turn detector, understands
// with a transcriber, thinks with a chat session, and speaks with a
// synthesizer, all concurrently: it keeps listening while it thinks and
// speaks, drops a reply when the speaker turns out not to be finished,
// stops within one frame when interrupted, and remembers only what was
// actually heard. The application sees no turns, only audio in and out:
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
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"
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
	preroll  = 300 * time.Millisecond  // audio kept before speech is detected: the detector fires late
	pause    = 200 * time.Millisecond  // silence after which the turn detector is asked, and again as it grows
	giveUp   = 1500 * time.Millisecond // silence that ends a turn whatever the detector says
	shortest = 300 * time.Millisecond  // speech an utterance needs
	bargeIn  = 300 * time.Millisecond  // speech over the agent that interrupts it; shorter is a backchannel
	resume   = 160 * time.Millisecond  // speech that shows a speaker was not finished after all
	// Text speed, for estimating how much of an interrupted reply was heard.
	charsPerSecond = 14
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
	// Voice selects the synthesizer's voice and language.
	Voice speech.SpeakOptions
	// Reply shapes the language model's answers.
	Reply chat.Options
	// Addressed, when not nil, decides whether an utterance is meant for
	// the agent, as in a meeting where people also talk to each other; it
	// runs on the transcript. Nil answers everything.
	Addressed func(text string) bool
	// OnText, when not nil, receives each transcribed utterance
	// (chat.User) and each reply once finished or interrupted
	// (chat.Assistant), from a worker goroutine.
	OnText func(role chat.Role, text string)
	// OnError, when not nil, receives errors of the workers.
	OnError func(error)
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
	speaker atomic.Uint32  // current reply generation; 0 when none

	mu    sync.Mutex
	state speech.DuplexState

	jobs   chan job
	cancel context.CancelFunc // cancels the current reply; guarded by mu
	busy   atomic.Bool        // a reply is being prepared or spoken
	stop   chan struct{}
	wg     sync.WaitGroup
	closed bool
}

// job is one reply to produce: an utterance to transcribe and answer, or
// text to say.
type job struct {
	audio []float32
	say   string
	gen   uint32
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
		play: newRing[float32](120 * out), jobs: make(chan job, 4), stop: make(chan struct{})}
	vad, err := gopus.NewVAD(inRate)
	if err != nil {
		return nil, err
	}
	c.wg.Add(2)
	go c.listen(vad)
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
	n := c.play.read(out)
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

// Say speaks text as soon as the current reply ends.
func (c *Cascade) Say(text string) error {
	select {
	case c.jobs <- job{say: text}:
		return nil
	default:
		return errors.New("duplex: too many replies queued")
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

// listen follows the room's audio frame by frame: voice activity marks
// speech, a pause asks the turn detector whether the speaker is done, and
// speech over the agent either lets it go on (an acknowledgement) or
// interrupts it.
func (c *Cascade) listen(vad *gopus.VAD) {
	defer c.wg.Done()
	var (
		frame      = make([]float32, inFrame)
		pcm16      = make([]int16, inFrame)
		keep       = int(preroll/frameTime) * inFrame
		early      = make([]float32, 0, keep)
		utterance  = make([]float32, 0, 60*inRate)
		talked     time.Duration
		silence    time.Duration
		overlap    time.Duration // continuous speech so far
		dispatched bool          // the utterance was handed to the responder
	)
	reset := func() { utterance, talked, silence, dispatched = utterance[:0], 0, 0, false }
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
			audible := c.busy.Load() && (c.played.Load() > 0 || c.play.len() > 0)
			switch {
			case dispatched && audible:
				// The agent is answering: the utterance is done.
				reset()
			case dispatched && c.busy.Load() && overlap >= resume:
				// Speech before the answer is heard: the speaker was not
				// finished. Drop the answer and hear the whole utterance.
				c.interrupt()
				dispatched = false
			}
			if audible && overlap >= bargeIn {
				c.interrupt() // a real interruption, not an acknowledgement
			}
			if len(utterance) == 0 && !speaking {
				early = append(early, frame...)
				if over := len(early) - keep; over > 0 {
					early = early[:copy(early, early[over:])]
				}
				continue
			}
			if len(utterance) == 0 {
				utterance = append(utterance, early...)
				early = early[:0]
			}
			if len(utterance)+inFrame <= cap(utterance) {
				utterance = append(utterance, frame...)
			}
			if speaking {
				talked, silence = talked+frameTime, 0
				continue
			}
			silence += frameTime
			if dispatched || silence%pause != 0 && silence < giveUp {
				if silence >= giveUp && !dispatched {
					reset()
				}
				continue
			}
			p, err := c.cfg.TurnDetector.PredictInto(utterance, inRate, 1)
			if err != nil {
				c.fail(err)
				continue
			}
			if !p.Complete && silence < giveUp {
				continue
			}
			if talked < shortest {
				reset() // a cough or a click
				continue
			}
			c.dispatch(job{audio: append([]float32(nil), utterance...)})
			dispatched = true
		}
	}
}

// dispatch replaces any reply in preparation with j.
func (c *Cascade) dispatch(j job) {
	c.interrupt()
	c.busy.Store(true)
	select {
	case c.jobs <- j:
	default:
		c.fail(errors.New("duplex: responder is behind; utterance dropped"))
	}
}

// respond produces replies one at a time: it transcribes, asks the chat
// session, and streams the answer's text into the synthesizer, whose
// audio queues for Step.
func (c *Cascade) respond() {
	defer c.wg.Done()
	var t speech.Transcript
	pieces := make(chan string, 256)
	for {
		var j job
		select {
		case <-c.stop:
			return
		case j = <-c.jobs:
		}
		// A newer job waiting supersedes this one.
		for len(c.jobs) > 0 {
			j = <-c.jobs
		}
		ctx, cancel := context.WithCancel(context.Background())
		c.mu.Lock()
		c.cancel = cancel
		c.mu.Unlock()
		c.played.Store(0)
		mark := c.cfg.Session.Checkpoint()
		text := j.say
		if j.audio != nil {
			if err := c.cfg.Transcriber.Transcribe(ctx, j.audio, speech.Options{}, &t); err != nil {
				c.finish(cancel, err)
				continue
			}
			text = strings.TrimSpace(string(t.Text))
			if text == "" || c.cfg.Addressed != nil && !c.cfg.Addressed(text) {
				if text != "" && c.cfg.OnText != nil {
					c.cfg.OnText(chat.User, text)
				}
				c.finish(cancel, nil)
				continue
			}
			if err := c.cfg.Session.Add(chat.User, text); err != nil {
				c.finish(cancel, err)
				continue
			}
		}
		// The synthesizer reads text as the model writes it.
		var reply strings.Builder
		done := make(chan error, 1)
		go func() {
			done <- c.cfg.Synthesizer.Speak(ctx, c.cfg.Voice, func() ([]byte, error) {
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
				c.play.write(pcm)
				return ctx.Err()
			})
		}()
		var err error
		if j.audio != nil {
			err = c.cfg.Session.Reply(ctx, c.cfg.Reply, func(p []byte) error {
				reply.Write(p)
				select {
				case pieces <- string(p):
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			})
		} else {
			reply.WriteString(text)
			pieces <- text
			c.cfg.Session.Add(chat.Assistant, text)
		}
		close(pieces)
		speakErr := <-done
		pieces = make(chan string, 256)
		interrupted := ctx.Err() != nil
		// Wait for the reply to be heard, or cut.
		for !interrupted && c.play.len() > 0 {
			select {
			case <-ctx.Done():
				interrupted = true
			case <-time.After(frameTime):
			}
		}
		said := reply.String()
		switch played := c.played.Load(); {
		case interrupted && played == 0:
			// Superseded before a sound: the conversation never had it.
			c.cfg.Session.Restore(mark)
			said, text = "", ""
		case interrupted:
			heard := int(time.Duration(played) * time.Second / time.Duration(c.outRate) * charsPerSecond / time.Second)
			if j.audio != nil {
				c.cfg.Session.Truncate(heard)
			}
			said = cut(said, heard) + "…"
		}
		if c.cfg.OnText != nil && j.audio != nil && text != "" {
			c.cfg.OnText(chat.User, text)
		}
		if c.cfg.OnText != nil && said != "" {
			c.cfg.OnText(chat.Assistant, said)
		}
		if interrupted {
			err, speakErr = nil, nil
		}
		c.finish(cancel, errors.Join(err, speakErr))
	}
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
