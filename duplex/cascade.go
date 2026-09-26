// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package duplex builds speech.Duplex agents from gophonic's lanes. A
// Cascade listens with a voice detector, understands with a transcriber,
// thinks with a chat session, and speaks with a synthesizer, all
// concurrently. It transcribes speech while it is spoken, keeping the
// conversation evaluated up to what has been said, and prepares its answer
// at the speaker's first pause, playing it the moment the turn is over, as
// the transcriber judges when it can (speech.Options.Turn) or else a turn
// detector: almost nothing is left to compute once the speaker is done. It
// drops the answer when the speaker goes on, stops within one frame when
// interrupted, and remembers only what was actually heard.
//
// The agent acts at moments: after each utterance, after each Note (a chat
// message, someone joining, a timer), when a pause it asked for is over,
// and, if Config.Idle is set, after a quiet spell. At every moment the
// model is asked what to say, and speech and silence are its only
// primitives: it answers with words, with <silent>, or with words that end
// in <silent 30s>, which asks to be asked again after that time, as when it
// must wait for something or remind someone. The application sees no
// turns, only audio in and out:
//
//	agent, err := duplex.New(duplex.Config{Prompt: "You are Gopher."}, asr, llm, tts)
//	for each 20 ms frame { state, err := agent.Step(in, out) }
//
// New finds the lanes it needs among the models it is given, by what they
// provide; any lane set in Config takes precedence, so a custom
// transcriber, detector, conversation, or voice composes in unchanged.
package duplex

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"
	"unsafe"

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
	// A transcriber that hears turns end judges the pause's utterance as it
	// transcribes it: a turn it is sure of ends at once; one it is not is
	// judged again from the pause's audio at each of rejudge, ending when
	// it is over, and a turn it heard as unfinished (under unsure) waits
	// up to giveUpUnfinished.
	sure     = 0.9
	unsure   = 0.1
	shortest = 300 * time.Millisecond // speech an utterance needs
	resume   = 160 * time.Millisecond // speech that shows a speaker was not finished after all
	// Speech this soon into the agent's answer continues the speaker's
	// turn: the answer stops and is forgotten, and the whole utterance is
	// heard again.
	resumeWindow = 800 * time.Millisecond
	// Speech over the agent is judged as its words arrive: once it has gone
	// on for overlapPeek, again after each overlapEvery more of it, and
	// when it pauses for overlapEnd. Words addressed to the agent stop it;
	// an acknowledgement lets it go on, and so, until the next judgment, do
	// words not yet made out: a slow speaker's first half second can hold
	// half a word. Talking over the agent for overlapLong stops it outright.
	overlapEnd   = 160 * time.Millisecond
	overlapPeek  = 560 * time.Millisecond
	overlapEvery = 240 * time.Millisecond
	overlapLong  = 1500 * time.Millisecond
	// firstAudio bounds how long the model waits, once it has written the
	// first words, for the voice to sound them: the listener waits for that
	// frame, not for the words after it, and until it is out the voice has
	// the GPU to itself. A voice that needs more text to start gets it then.
	firstAudio = 300 * time.Millisecond
	// Transcription while speaking: the speech an utterance needs before it
	// is first transcribed, and the new speech that starts another pass.
	scribeFirst = inRate / 2
	longest     = 60 * inRate // samples of an utterance heard; more is dropped
	scribeStep  = inRate * 160 / 1000
	// maxToolRounds bounds the replies one moment gets while tools are
	// called.
	maxToolRounds = 4
)

// Config shapes a Cascade. Every field is optional.
type Config struct {
	// Prompt is the system prompt of a conversation New starts;
	// MomentsPrompt follows it.
	Prompt string

	// The lanes. New opens those left nil from its models; the Cascade
	// owns all of them and closes them in Close.
	Transcriber speech.Transcriber
	Turns       speech.TurnDetector // judges turns when the transcriber does not
	Session     chat.Session        // the conversation, its system prompt added
	Voice       speech.Synthesizer

	// Listen configures transcription, such as the languages spoken.
	Listen speech.Options
	// Speak selects the voice, its language, and how it speaks.
	Speak speech.SpeakOptions
	// Reply shapes the language model's answers.
	Reply chat.Options
	// Tools are functions the agent may call as part of a reply. New
	// offers them to the conversation it starts; a Session given here must
	// have been started with their specs (chat.Specs).
	Tools []chat.Tool

	// Idle is how long the agent may hear nothing, and be told nothing,
	// before it is asked once whether it has something to say; zero never
	// asks.
	Idle time.Duration

	// Interruptions judges speech over the agent's voice: given the text
	// "Assistant: <what it was saying>\nUser: <what was said over it>", its
	// first label means go on and its second means stop. Nil uses a zero-
	// shot classifier from New's models when one provides it, and
	// otherwise a lexicon of acknowledgements and stop words.
	Interruptions speech.TextClassifier
	// Quiet judges whether what was said asks the agent to be quiet, to
	// stop talking, or to wait: its second label means it does. A silence
	// the model chooses when nothing asked for it (alone with one person,
	// any silence; in a meeting, one until something happens) is
	// overruled: the model answers after all, told why. Nil uses a zero-
	// shot classifier from New's models when one provides it, and
	// otherwise lets every silence stand.
	Quiet speech.TextClassifier
	// Wake judges, while the agent keeps a silence it chose until
	// something happens, whether what was just said is it: given the text
	// "Silent until: <what>\nUser: <what was said>", its second label means
	// the silence is over, and the model answers, told so; otherwise the
	// words join the conversation unanswered. Nil uses a zero-shot
	// classifier from New's models when one provides it; otherwise, and
	// once a silence is MaxSilence old, the model judges each moment
	// itself, reminded of what it waits for.
	Wake speech.TextClassifier

	// Observer sees what is heard and said, each reply's stages, and
	// errors, from worker goroutines. Nil observes nothing.
	Observer Observer
}

// MomentsPrompt ends the system prompt of a conversation New starts: the
// grammar of speech and silence.
const MomentsPrompt = `You are asked what to say at moments: after someone speaks, when something is noted, and when you asked to be. Reply with your words, or with exactly <silent> to say nothing, as when what was said is clearly meant for someone else. A greeting, a single word, or anything unclear or misheard gets a short reply. A note is information, not a question: unless it asks you something, reply <silent>. A tool's result is answered, in words. Asked to be quiet until something happens, reply <silent until "hi">, naming what ends it, and then <silent> to everything until it happens. To be asked again after some time, as when you must wait or remind someone, end your reply with <silent 30s> (any number of seconds or minutes). Never repeat or read back what someone said: everyone heard it.`

// MaxSilence is how long a silence the model chose until something
// happens is judged by Config.Wake alone; after it, the model judges each
// moment itself.
const MaxSilence = 2 * time.Minute

const silentMark = "<silent"

// Cascade is a speech.Duplex made of lanes; see the package comment.
type Cascade struct {
	cfg              Config
	obs              Observer
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

	mu      sync.Mutex
	state   speech.DuplexState
	speaker string             // who is speaking now, as the application says; guarded by mu
	notes   []string           // messages to add at the next moment; guarded by mu
	cancel  context.CancelFunc // cancels the current reply; guarded by mu

	asr        sync.Mutex // the transcriber serves the scribe, listener, and responder
	probs      []float32  // Interruptions' output
	quietProbs []float32  // Quiet's output
	wakeProbs  []float32  // Wake's output

	// The utterance being heard, written by the listener and transcribed
	// by the scribe as it grows; partial is the scribe's latest transcript
	// of it.
	utt       utterance
	uttReady  chan struct{}
	hearing   atomic.Bool   // speech is being heard: moments wait for it
	pauses    atomic.Uint32 // speaker pauses so far: each aborts the scribe's pass
	pass      passContext
	partialMu sync.Mutex
	partial   speech.Transcript
	partialOf uint32 // the utterance generation partial transcribes
	partialN  int    // its samples
	prefill   chan struct{}
	// Judgments of the pending answer's utterance: the transcriber's, when
	// it hears turns end (hears), and the language model's log10
	// probability that the words end the speaker's message.
	hears    bool
	wordsMu  sync.Mutex
	turnJob  uint32
	turn     speech.Prediction
	wordsJob uint32
	words    float32

	jobs      chan job
	noted     chan struct{} // signals notes, or a moment due
	momentDue atomic.Bool   // a note moment waits for the speaker to finish
	busy      atomic.Bool   // a reply is being prepared or spoken
	stop      chan struct{}
	wg        sync.WaitGroup
	closed    bool

	// Moments the agent plans (<silent 30s>) and idle spells, and the
	// responder's clock while a reply plays.
	planned *time.Timer
	idle    *time.Timer
	tick    *time.Ticker
	// The current reply's context, for the responder's writer.
	replyCtx context.Context

	// The responder's reply: the model's text, the speakable text sent to
	// the voice (and captioned), and the writer that parses the reply as
	// it is written.
	reply       []byte
	said        []byte
	saying      []byte // said, for judging speech over it; guarded by mu
	rw          replyWriter
	noteBuf     []byte
	lastSpeaker string
	// silentUntil is what the model named as ending a silence it chose,
	// and silentSince when; each later moment is judged against it until
	// the model speaks.
	silentUntil string
	silentSince time.Time
	t, partialT speech.Transcript
	// The voice. The responder writes said to it as it grows (voiceAt bytes
	// so far; voicing once it began an utterance), and spans maps each
	// length of said to the length of reply it was spoken from. The pump
	// reads the utterance's audio into play.
	voicing   bool
	voiceAt   int
	spans     []span
	firstSent time.Time
	pumpJobs  chan job
	voiceEv   chan struct{} // the voice's first audio is out, or it ended
	speakDone chan error
	sounded   atomic.Bool
	captioned int // bytes of said captioned as voiced
	userShown bool
}

// span is a length of said and the length of the reply it came from.
type span struct{ said, reply int }

// job is one moment: an utterance to transcribe and answer, text to say,
// or a moment with nothing new but the notes. A held job plays only once
// its turn is released.
type job struct {
	audio   []float32
	say     string
	speaker string
	utt     uint32    // the utterance generation of audio
	id      uint32    // set by dispatch
	held    bool      // prepared at a pause, before the turn is known to be over
	at      time.Time // the speaker's pause
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
// speech.TurnDetector unless the transcriber judges turns itself, a
// chat.Generator for the conversation, and a speech.Synthesizer.
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
	if err := open(&cfg.Voice, models, &opened); err != nil {
		return nil, err
	}
	if cfg.Interruptions == nil || cfg.Quiet == nil || cfg.Wake == nil {
		var z speech.ZeroShot
		if open(&z, models, &opened) == nil {
			for _, q := range []struct {
				c        *speech.TextClassifier
				question string
				labels   []string
			}{{&cfg.Interruptions, interruptQuestion, interruptLabels}, {&cfg.Quiet, quietQuestion, quietLabels}, {&cfg.Wake, wakeQuestion, wakeLabels}} {
				if *q.c != nil {
					continue
				}
				if *q.c, err = z.Classifier(q.question, q.labels); err != nil {
					return nil, err
				}
				opened = append(opened, *q.c)
			}
		}
	}
	if cfg.Session == nil {
		var g chat.Generator
		if err := open(&g, models, &opened); err != nil {
			return nil, err
		}
		prompt := MomentsPrompt
		if cfg.Prompt != "" {
			prompt = cfg.Prompt + "\n" + MomentsPrompt
		}
		if cfg.Session, err = g.NewSession(prompt, chat.Specs(cfg.Tools)...); err != nil {
			return nil, err
		}
		opened = append(opened, cfg.Session)
	}
	if cfg.Observer == nil {
		cfg.Observer = Base{}
	}
	out := cfg.Voice.SampleRate()
	c := &Cascade{cfg: cfg, obs: cfg.Observer, outRate: out, outSize: out / 50,
		in: newRing[float32](2 * inRate), inReady: make(chan struct{}, 1),
		play: newRing[float32](120 * out), release: make(chan struct{}, 1),
		utt: utterance{pcm: make([]float32, 0, longest)}, uttReady: make(chan struct{}, 1), prefill: make(chan struct{}, 1),
		jobs: make(chan job, 4), noted: make(chan struct{}, 1), stop: make(chan struct{}),
		probs: make([]float32, 2), quietProbs: make([]float32, 2), wakeProbs: make([]float32, 2),
		pumpJobs: make(chan job, 1), voiceEv: make(chan struct{}, 1), speakDone: make(chan error, 1)}
	c.pass = passContext{Context: context.Background(), c: c}
	c.rw.c = c
	c.planned = time.AfterFunc(time.Hour, c.plannedMoment)
	c.planned.Stop()
	c.idle = time.AfterFunc(time.Hour, c.idleMoment)
	c.idle.Stop()
	c.tick = time.NewTicker(frameTime)
	vad, err := gopus.NewVAD(inRate)
	if err != nil {
		return nil, err
	}
	if err := c.warm(); err != nil {
		return nil, err
	}
	if !c.hears {
		if err := open(&c.cfg.Turns, models, &opened); err != nil {
			return nil, err
		}
	}
	c.wg.Add(4)
	go c.listen(vad)
	go c.scribe()
	go c.respond()
	go c.pump()
	c.active()
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

// warm runs the transcriber and the synthesizer once, and evaluates the
// system prompt, so that the first turn is as fast as every later one.
func (c *Cascade) warm() error {
	if err := c.cfg.Session.Prefill(context.Background()); err != nil {
		return err
	}
	var t speech.Transcript
	opts := c.cfg.Listen
	opts.Turn = true
	err := c.cfg.Transcriber.Transcribe(context.Background(), make([]float32, inRate), opts, &t)
	c.hears = err == nil
	if errors.Is(err, speech.ErrUnsupported) {
		err = c.cfg.Transcriber.Transcribe(context.Background(), make([]float32, inRate), c.cfg.Listen, &t)
	}
	if err != nil {
		return err
	}
	v := c.cfg.Voice
	if err := v.Begin(context.Background(), c.cfg.Speak); err != nil {
		return err
	}
	v.Write([]byte("Hi."))
	v.End()
	for pcm := make([]float32, c.outSize); ; {
		if _, err := v.Read(pcm); errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return err
		}
	}
}

// Rates: 16 kHz in, the synthesizer's rate out.
func (c *Cascade) Rates() (in, out int) { return inRate, c.outRate }

// Frame is 20 ms.
func (c *Cascade) Frame() (in, out int) { return inFrame, c.outSize }

// Step hands in to the listener and writes the next 20 ms of speech. A nil
// in is a gap, as when a packet is late: not silence.
func (c *Cascade) Step(in, out []float32) (speech.DuplexState, error) {
	if in != nil && len(in) != inFrame || len(out) != c.outSize {
		return speech.Listening, errors.New("duplex: Step takes one frame in and out")
	}
	if in != nil {
		c.in.write(in)
		select {
		case c.inReady <- struct{}{}:
		default:
		}
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
	return c.state, nil
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

// Note tells the agent that something happened, in words: a participant's
// chat message, someone joining, a tool's result. It joins the
// conversation as a system message at the next moment, which it brings
// about: the agent may answer it, or keep silent.
func (c *Cascade) Note(text string) error { return c.note(text, true) }

// note queues text; activity resets the idle timer.
func (c *Cascade) note(text string, activity bool) error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return chat.ErrClosed
	}
	c.notes = append(c.notes, text)
	c.mu.Unlock()
	if activity {
		c.active()
	}
	select {
	case c.noted <- struct{}{}:
	default:
	}
	return nil
}

// Speaker names who is speaking now, so that the conversation knows whose
// words it hears in a call with several people. It allocates nothing and
// may be called every frame; an empty name means no one in particular.
func (c *Cascade) Speaker(name string) {
	c.mu.Lock()
	c.speaker = name
	c.mu.Unlock()
}

// active notes activity: an idle spell starts over.
func (c *Cascade) active() {
	if c.cfg.Idle > 0 {
		c.idle.Reset(c.cfg.Idle)
	}
}

// idleMoment asks the agent, once, whether it has something to say after
// a quiet spell.
func (c *Cascade) idleMoment() {
	c.note("Nothing has happened for a while.", false)
}

// plannedMoment tells the agent the time it asked for has passed.
func (c *Cascade) plannedMoment() {
	c.note("The time you asked to wait has passed.", false)
}

// addNotes adds the messages waiting to the conversation, reporting
// whether there were any.
func (c *Cascade) addNotes() bool {
	c.mu.Lock()
	notes := c.notes
	c.notes = nil
	c.mu.Unlock()
	for _, n := range notes {
		if err := c.cfg.Session.Add(chat.System, n); err != nil {
			c.fail(err)
		}
	}
	return len(notes) > 0
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
	c.planned.Stop()
	c.idle.Stop()
	c.tick.Stop()
	close(c.stop)
	c.wg.Wait()
	var turns, judge, quiet, wake error
	if c.cfg.Turns != nil {
		turns = c.cfg.Turns.Close()
	}
	if c.cfg.Interruptions != nil {
		judge = c.cfg.Interruptions.Close()
	}
	if c.cfg.Quiet != nil {
		quiet = c.cfg.Quiet.Close()
	}
	if c.cfg.Wake != nil {
		wake = c.cfg.Wake.Close()
	}
	return errors.Join(c.cfg.Transcriber.Close(), turns, judge, quiet, wake, c.cfg.Session.Close(), c.cfg.Voice.Close())
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
		judged   time.Duration // talk over the agent when it was last judged
		paused   time.Time     // when the speaker last paused
		heardAt  time.Time     // when the agent's voice became audible
		pending  uint32        // the answer prepared at the pause, held until the turn ends
		answered bool          // the turn ended and its answer was released
		stopped  bool          // speech over the agent stopped it
	)
	reset := func() {
		c.utt.reset()
		c.hearing.Store(false)
		talked, silence, judged, pending, answered = 0, 0, 0, 0, false
		if c.momentDue.Load() {
			select {
			case c.noted <- struct{}{}:
			default:
			}
		}
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
				c.hearing.Store(true)
				c.active()
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
				// Speech over the agent is judged as its words arrive and
				// when it pauses, or stops the agent if it goes on; turns
				// wait until the agent is silent.
				switch {
				case talked >= overlapLong:
					c.interrupt()
					c.obs.Stage(Interrupted, talked)
					stopped = true
				case speaking && talked >= overlapPeek && talked >= judged+overlapEvery:
					// Long enough to be more than an acknowledgement, or
					// longer by some words: judge what has been said so far.
					judged = talked
					if c.interrupts(utterance) {
						c.interrupt()
						c.obs.Stage(Interrupted, talked)
						stopped = true
					}
				case talked > 0 && silence == overlapEnd:
					if c.interrupts(utterance) {
						c.interrupt()
						c.obs.Stage(Interrupted, talked)
						stopped = true
					} else {
						c.obs.Stage(Continued, talked)
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
			if c.hears {
				p, known := c.turnOf(pending)
				if known && p.Probability < sure && slices.Contains(rejudge[:], silence) {
					p, known = c.judgeTurn(pending, utterance)
				}
				if !heardOver(silence, p, known) {
					continue
				}
			} else {
				if silence%check != 0 && silence < giveUp {
					continue
				}
				p, err := c.cfg.Turns.Predict(utterance, inRate, 1)
				if err != nil {
					c.fail(err)
					continue
				}
				words, known := c.wordsOf(pending)
				if !turnOver(silence, p, words, known) {
					continue
				}
			}
			if talked < shortest {
				reset() // a cough or a click
				continue
			}
			if pending == 0 {
				paused = time.Now()
				pending = c.dispatch(job{audio: append([]float32(nil), utterance...), utt: c.utt.gen, at: paused})
			}
			c.obs.Stage(TurnOver, silence) // after the speaker stopped
			c.releaseTurn(pending)
			c.hearing.Store(false)
			c.active()
			pending, answered = 0, true
		}
	}
}

// rejudge are the pauses at which a turn the transcriber was unsure of is
// judged again, hearing the pause itself.
var rejudge = [...]time.Duration{240 * time.Millisecond, 600 * time.Millisecond}

// heardOver decides whether a pause of silence ends the speaker's turn,
// from the transcriber's judgment of it, when known.
func heardOver(silence time.Duration, p speech.Prediction, known bool) bool {
	switch {
	case !known:
		return silence >= giveUp
	case p.Probability >= sure:
		return true
	case p.Complete && silence >= rejudge[0]:
		return true
	case p.Probability >= unsure:
		return silence >= giveUp
	default:
		return silence >= giveUpUnfinished
	}
}

// judgeTurn transcribes the utterance heard so far, pause included, and
// records the transcriber's judgment of its turn for job id.
func (c *Cascade) judgeTurn(id uint32, audio []float32) (speech.Prediction, bool) {
	var t, partial speech.Transcript
	opts := c.cfg.Listen
	opts.Turn = true
	if c.partialFor(c.utt.gen, &partial) {
		opts.Partial = &partial
	}
	c.asr.Lock()
	err := c.cfg.Transcriber.Transcribe(context.Background(), audio, opts, &t)
	c.asr.Unlock()
	if err != nil {
		c.fail(err)
		return c.turnOf(id)
	}
	c.wordsMu.Lock()
	c.turnJob, c.turn = id, t.Turn
	c.wordsMu.Unlock()
	return t.Turn, true
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

// turnOf returns the transcriber's judgment of job id's turn, once the
// responder has it.
func (c *Cascade) turnOf(id uint32) (speech.Prediction, bool) {
	c.wordsMu.Lock()
	defer c.wordsMu.Unlock()
	return c.turn, id != 0 && c.turnJob == id
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
			c.mu.Lock()
			speaker := c.speaker
			c.mu.Unlock()
			c.obs.Heard(speaker, prev.Text, false)
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

// prefillPartial judges what the speaker has said so far, as the answer
// will: that evaluates the conversation up to it, and when the utterance
// ends as it was heard, the answer's judgment is already made.
func (c *Cascade) prefillPartial() {
	c.partialMu.Lock()
	text := unclosed(bytes.TrimSpace(c.partial.Text))
	c.partialMu.Unlock()
	if len(text) == 0 {
		return
	}
	if _, err := c.cfg.Session.Finished(context.Background(), chat.User, view(text)); err != nil {
		c.fail(err)
	}
}

// view returns b as a string without copying, for calls that do not
// retain it.
func view(b []byte) string { return unsafe.String(unsafe.SliceData(b), len(b)) }

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
// dropped first. A job with no turn (a note, a planned moment, text to
// say) waits for nothing.
func (c *Cascade) awaitTurn(ctx context.Context, id uint32) bool {
	if id == 0 {
		return ctx.Err() == nil
	}
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
// stop: acknowledgements, noise, and words not yet made out do not; stop
// words do; anything else is judged in the context of what the agent is
// saying.
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
	return c.judge(normalize(string(t.Text)), string(t.Text))
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
	saying := string(c.saying)
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

// appendSpeakable appends src to dst without what a voice cannot say and
// captions need not show: emoji and other pictographs, and markdown
// emphasis and headings.
func appendSpeakable(dst, src []byte) []byte {
	for len(src) > 0 {
		r, size := utf8.DecodeRune(src)
		src = src[size:]
		switch {
		case r == '*' || r == '_' || r == '#' || r == '`' || r == '~':
			continue
		case r == 0x200d || r >= 0xfe00 && r <= 0xfe0f: // joiners and variation selectors
			continue
		case unicode.Is(unicode.So, r) || unicode.Is(unicode.Sk, r) || r >= 0x1f000:
			continue
		}
		dst = utf8.AppendRune(dst, r)
	}
	return dst
}

// unclosed drops closing punctuation: a transcript of it alone is empty.
func unclosed(text []byte) []byte {
	return bytes.TrimRightFunc(text, func(r rune) bool { return unicode.IsPunct(r) || unicode.IsSpace(r) })
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
	c.mu.Lock()
	j.speaker = c.speaker
	c.mu.Unlock()
	c.held.Store(j.held)
	c.busy.Store(true)
	// A job waiting unanswered is superseded by this one, as the responder
	// would supersede it; so a full queue loses its oldest job, not this.
	for {
		select {
		case c.jobs <- j:
			return j.id
		default:
		}
		select {
		case <-c.jobs:
		default:
		}
	}
}

// respond produces replies one at a time: it transcribes, asks the chat
// session, and streams the answer's text into the synthesizer, whose
// audio queues for Step. Between jobs it adds notes, and answers them
// once the speaker is done.
func (c *Cascade) respond() {
	defer c.wg.Done()
	for {
		select {
		case <-c.stop:
			return
		case <-c.noted:
			if c.addNotes() {
				c.momentDue.Store(true)
			}
		case <-c.prefill:
			c.prefillPartial()
			continue
		case j := <-c.jobs:
			// A newer job waiting supersedes this one.
			for len(c.jobs) > 0 {
				j = <-c.jobs
			}
			c.run(j)
		}
		if c.momentDue.Load() && !c.hearing.Load() && len(c.jobs) == 0 {
			c.momentDue.Store(false)
			c.busy.Store(true)
			c.run(job{})
		}
	}
}

// run produces the reply of one job.
func (c *Cascade) run(j job) {
	c.addNotes()
	c.momentDue.Store(false) // this reply answers the notes too
	c.resumed.Store(false)
	if !j.held {
		c.held.Store(false)
	}
	ctx, cancel := context.WithCancel(context.Background())
	c.replyCtx = ctx
	c.mu.Lock()
	c.cancel = cancel
	c.mu.Unlock()
	if j.at.IsZero() {
		j.at = time.Now()
	}
	c.played.Store(0)
	mark := c.cfg.Session.Checkpoint()
	var text []byte
	if j.audio != nil {
		opts := c.cfg.Listen
		opts.Turn = c.hears
		if c.partialFor(j.utt, &c.partialT) {
			opts.Partial = &c.partialT
		}
		c.asr.Lock()
		err := c.cfg.Transcriber.Transcribe(ctx, j.audio, opts, &c.t)
		c.asr.Unlock()
		if err != nil {
			if ctx.Err() != nil {
				err = nil // superseded
			}
			c.finish(cancel, err)
			return
		}
		if c.hears {
			c.wordsMu.Lock()
			c.turnJob, c.turn = j.id, c.t.Turn
			c.wordsMu.Unlock()
		}
		c.obs.Stage(Transcribed, time.Since(j.at))
		text = bytes.TrimSpace(c.t.Text)
		if len(unclosed(text)) == 0 {
			c.finish(cancel, nil)
			return
		}
		if j.speaker != "" && j.speaker != c.lastSpeaker {
			// Whose words these are, once, when the speaker changes.
			c.noteBuf = append(append(c.noteBuf[:0], j.speaker...), " is speaking."...)
			if err := c.cfg.Session.Add(chat.System, view(c.noteBuf)); err != nil {
				c.finish(cancel, err)
				return
			}
			c.lastSpeaker = j.speaker
		}
		skip, err := c.judgeSilence(ctx, text)
		if err != nil {
			c.finish(cancel, err)
			return
		}
		// Asked to be quiet, in so many words, the model is told so before
		// it answers: its silence then names what ends it.
		if !skip && c.cfg.Quiet != nil && quietWords.Match(text) && c.asksQuiet(ctx, text) {
			if err := c.cfg.Session.Add(chat.System, askedQuiet); err != nil {
				c.finish(cancel, err)
				return
			}
		}
		// Whether the words are finished helps judge the turn; their
		// evaluation is the reply's first step anyway.
		if p, err := c.cfg.Session.Finished(ctx, chat.User, view(text)); err == nil {
			c.wordsMu.Lock()
			c.wordsJob, c.words = j.id, float32(math.Log10(max(float64(p), 1e-30)))
			c.wordsMu.Unlock()
			c.obs.Stage(Judged, time.Since(j.at))
		} else if ctx.Err() == nil {
			c.fail(err)
		}
		if err := c.cfg.Session.Add(chat.User, view(text)); err != nil {
			c.finish(cancel, err)
			return
		}
		if skip {
			// Not what the silence waits for: the words join the
			// conversation and show once the turn is over, unanswered.
			if c.awaitTurn(ctx, j.id) {
				c.obs.Heard(j.speaker, text, true)
				c.obs.Stage(Silent, time.Since(j.at))
				if err := c.cfg.Session.Prefill(ctx); err != nil && ctx.Err() == nil {
					c.fail(err)
				}
			} else {
				c.cfg.Session.Restore(mark)
			}
			c.finish(cancel, nil)
			return
		}
	}
	// The voice reads the reply as the model writes it.
	c.reply, c.said = c.reply[:0], c.said[:0]
	c.mu.Lock()
	c.saying = c.saying[:0]
	c.mu.Unlock()
	c.rw.begin(j, text)
	c.captioned, c.userShown = 0, false
	c.sounded.Store(false)
	c.voicing, c.voiceAt, c.spans = false, 0, c.spans[:0]
	var err error
	if j.say != "" {
		c.reply = append(c.reply, j.say...)
		c.said = append(c.said, j.say...)
		c.spans = append(c.spans, span{len(c.said), len(c.reply)})
		err = c.voice()
		c.cfg.Session.Add(chat.Assistant, j.say)
	} else {
		if j.audio == nil {
			if err := c.remindSilence(); err != nil {
				c.finish(cancel, err)
				return
			}
		}
		replyMark := c.cfg.Session.Checkpoint()
		for overruled, retried := false, false; ; {
			// A reply may call tools; their results have the agent
			// answer again, on the same voice.
			called := false
			for round := 0; ; round++ {
				c.rw.round = len(c.reply)
				err = c.cfg.Session.Reply(ctx, c.cfg.Reply, &c.rw)
				if errors.Is(err, errReplyEnded) {
					err = nil
				}
				if err != nil || round+1 == maxToolRounds || !c.runCalls(ctx, j) {
					break
				}
				called = true
			}
			if ferr := c.rw.flush(); err == nil && ctx.Err() == nil {
				err = ferr
			}
			if len(bytes.TrimSpace(c.said)) > 0 || err != nil || ctx.Err() != nil {
				break
			}
			// A silent reply may be wrong twice over: a tool's result is
			// answered, not noted; and a silence answers only a request
			// for quiet (alone with one person, any silence; in a
			// meeting, one until something happens), so one chosen for a
			// misheard word is overruled. A silence that plans its next
			// moment (<silent 30s>, a reminder) ends by itself and stands.
			// Either way the model answers after all, told why.
			var note string
			switch {
			case called && !retried:
				note, retried = answerTools, true
			case j.audio != nil && !overruled && c.rw.wait == 0 && (c.rw.until != "" || j.speaker == "") && !c.asksQuiet(ctx, text):
				note, overruled = notAskedQuiet, true
				c.obs.Stage(Overruled, time.Since(j.at))
				err = c.cfg.Session.Restore(replyMark)
			}
			if note == "" || err != nil {
				break
			}
			if err = c.cfg.Session.Add(chat.System, note); err != nil {
				break
			}
			c.reply, c.said, c.spans = c.reply[:0], c.said[:0], c.spans[:0]
			c.rw.begin(j, text)
		}
		c.said = bytes.TrimRightFunc(c.said, unicode.IsSpace) // before a marker
		if c.rw.echoed && err == nil && ctx.Err() == nil && len(c.cfg.Session.Calls()) == 0 {
			// The conversation keeps the reply as said, so that the
			// model never sees itself repeating.
			c.cfg.Session.Restore(replyMark)
			c.cfg.Session.Add(chat.Assistant, view(c.said))
			c.obs.Stage(Dropped, time.Since(j.at))
		}
		if err == nil && ctx.Err() == nil {
			if len(bytes.TrimSpace(c.said)) == 0 {
				c.obs.Stage(Silent, time.Since(j.at))
			}
			if c.rw.wait > 0 {
				c.obs.Stage(Planned, c.rw.wait)
			}
		}
	}
	var speakErr error
	if c.voicing {
		c.cfg.Voice.End()
		speakErr = <-c.speakDone
	}
	silent := len(bytes.TrimSpace(c.said)) == 0
	if silent && j.audio != nil && c.awaitTurn(ctx, j.id) {
		// Silence, once the turn is over, shows only what was said.
		c.obs.Heard(j.speaker, text, true)
	}
	interrupted := c.waitPlayback(ctx, func() { c.caption(j, text) })
	if j.audio != nil && !c.userShown && len(text) > 0 && c.played.Load() > 0 && !c.resumed.Load() {
		// A reply heard only after the model finished writing it.
		c.obs.Heard(j.speaker, text, true)
	}
	final := true
	switch played := c.played.Load(); {
	case interrupted && played == 0:
		// Superseded before a sound: the conversation never had it.
		c.cfg.Session.Restore(mark)
		final = false
	case interrupted && c.resumed.Load():
		// Cut as it began by the speaker going on: the utterance is
		// heard again whole, so the conversation and the captions
		// forget both.
		c.cfg.Session.Restore(mark)
		final = false
	case interrupted:
		heard := min(len(c.said), c.voicedAt(played))
		if j.audio != nil {
			// The conversation keeps what was heard: of the reply it holds,
			// or of said when the reply was replaced by it.
			n := heard
			if !c.rw.echoed {
				n = max(0, c.replyOf(heard)-c.rw.round)
			}
			c.cfg.Session.Truncate(n)
		}
		c.said = append(c.said[:cut(c.said, heard)], "…"...)
	}
	if final && !silent {
		c.obs.Said(c.said, len(c.said), true)
	}
	// A reply that stands, and only one, changes the silence: the
	// conversation forgets a superseded one.
	if final && err == nil {
		switch {
		case !silent:
			c.silentUntil = "" // speaking ends a silence
		case c.rw.until != "":
			c.silentUntil, c.silentSince = c.rw.until, time.Now()
		}
	}
	if !interrupted && err == nil && c.rw.wait > 0 {
		c.planned.Reset(c.rw.wait)
	}
	if interrupted {
		err, speakErr = nil, nil
	}
	c.finish(cancel, errors.Join(err, speakErr))
}

// The zero-shot question that judges whether what was said asks the agent
// to be quiet, and the note that overrules a silence chosen otherwise.
const quietQuestion = `A voice assistant hears this in a call. Is the user asking it to be quiet, to stop talking, or to wait?`

var quietLabels = []string{
	"Something else, or it is unclear",
	"A request to be quiet, stop talking, or wait",
}

const notAskedQuiet = "No one asked you to be quiet: answer what was just said."

// answerTools is the note after a silent reply to a tool's result.
const answerTools = "Answer now, in words, with what the tool found."

// The zero-shot question that judges whether a silence the agent chose is
// over, and the note that tells the model so.
const wakeQuestion = `A voice assistant chose to stay silent until something happens. Does what the user just said end its silence?`

var wakeLabels = []string{
	"No: the assistant stays silent",
	"Yes: it happened, or the user tells the assistant it may speak again",
}

// wakes reports whether text ends the silence the agent keeps, as Wake
// judges it.
func (c *Cascade) wakes(ctx context.Context, text []byte) bool {
	c.noteBuf = append(append(append(append(c.noteBuf[:0], "Silent until: "...), c.silentUntil...), "\nUser: "...), text...)
	if err := c.cfg.Wake.ClassifyInto(ctx, view(c.noteBuf), c.wakeProbs); err != nil {
		if ctx.Err() == nil {
			c.fail(err)
		}
		return true
	}
	return c.wakeProbs[1] > c.wakeProbs[0]
}

// judgeSilence decides, while the model keeps a silence it chose until
// something happens, what to do with the words just heard. With Wake, and
// while the silence is younger than MaxSilence, Wake judges: words that
// end the silence are answered, the model told so, and others join the
// conversation unanswered (skip). Otherwise the model judges for itself,
// reminded of what it waits for.
func (c *Cascade) judgeSilence(ctx context.Context, text []byte) (skip bool, err error) {
	if c.silentUntil == "" {
		return false, nil
	}
	if c.cfg.Wake != nil && time.Since(c.silentSince) < MaxSilence {
		if !c.wakes(ctx, text) {
			return true, nil
		}
		c.noteBuf = append(append(append(c.noteBuf[:0], `What you were waiting for ("`...), c.silentUntil...), `") has happened: answer now.`...)
		return false, c.cfg.Session.Add(chat.System, view(c.noteBuf))
	}
	return false, c.remindSilence()
}

// askedQuiet is the note before a reply to a request for quiet, and
// quietWords the words that make the request worth judging.
const askedQuiet = `You are asked to be quiet: reply with nothing but <silent until "...">, naming what ends the silence.`

var quietWords = regexp.MustCompile(`(?i)\b(quiet|silent|silence|stop talking|stop speaking|shut up|wait|hold on|hang on|pause|don't talk|do not talk|hush|cala|silêncio|espera|quieto|calado)\b`)

// quietOnly matches a message that is nothing but a command to be quiet.
var quietOnly = regexp.MustCompile(`(?i)^\W*(wait|stop|quiet|be quiet|silence|shut up|hold on|hang on|pause|hush|one moment|just a moment|espera|cala-te|calado|silêncio)\W*$`)

// asksQuiet reports whether message asks the agent to be quiet, and so
// whether a silence it chose stands: a bare command ("Wait.", "Quiet!")
// does, and Quiet judges the rest.
func (c *Cascade) asksQuiet(ctx context.Context, message []byte) bool {
	if quietOnly.Match(message) {
		return true
	}
	if c.cfg.Quiet == nil {
		return true
	}
	if err := c.cfg.Quiet.ClassifyInto(ctx, view(message), c.quietProbs); err != nil {
		if ctx.Err() == nil {
			c.fail(err)
		}
		return true
	}
	return c.quietProbs[1] > c.quietProbs[0]
}

// remindSilence adds, while the model keeps a silence it chose until
// something happens, a note of what it is waiting for, so that it judges
// each moment against it.
func (c *Cascade) remindSilence() error {
	if c.silentUntil == "" {
		return nil
	}
	c.noteBuf = append(append(append(c.noteBuf[:0], `You are staying silent until "`...), c.silentUntil...),
		`" happens: reply <silent> unless it has.`...)
	return c.cfg.Session.Add(chat.System, view(c.noteBuf))
}

// pump plays each utterance the responder begins: it reads the voice into
// play as the audio is decoded, and reports when the utterance ends. Audio
// read after an interruption never plays.
func (c *Cascade) pump() {
	defer c.wg.Done()
	pcm := make([]float32, c.outSize)
	for {
		var j job
		select {
		case <-c.stop:
			return
		case j = <-c.pumpJobs:
		}
		gen, read := c.play.generation(), 0
		var err error
		for {
			var n int
			n, err = c.cfg.Voice.Read(pcm)
			if n > 0 {
				if read == 0 {
					c.obs.Stage(FirstAudio, time.Since(j.at))
				}
				read += n
				c.play.writeAt(gen, pcm[:n])
				if !c.sounded.Load() {
					c.sounded.Store(true)
					c.signalVoice()
				}
			}
			if err != nil {
				break
			}
		}
		if errors.Is(err, io.EOF) {
			err = nil
		}
		c.sounded.Store(true)
		c.signalVoice()
		c.speakDone <- err
	}
}

// waitPlayback waits for the reply to be heard, or cut, showing its
// captions as it plays, and reports whether it was cut. interrupt cancels
// the reply before it empties playback, so a tick can see playback empty
// without having seen the cancellation: that is a cut too.
func (c *Cascade) waitPlayback(ctx context.Context, caption func()) bool {
	for ctx.Err() == nil && c.play.len() > 0 {
		select {
		case <-ctx.Done():
		case <-c.tick.C:
			caption()
		}
	}
	return ctx.Err() != nil
}

// voicedAt reports how many bytes of said the voice has spoken by the
// played-th sample of the reply.
func (c *Cascade) voicedAt(played int64) int {
	if !c.voicing {
		return 0
	}
	return c.cfg.Voice.Voiced(int(played))
}

// replyOf returns the length of the reply that the first n bytes of said
// were spoken from, to the piece.
func (c *Cascade) replyOf(n int) int {
	r := 0
	for _, sp := range c.spans {
		if sp.said > n {
			break
		}
		r = sp.reply
	}
	return r
}

func (c *Cascade) signalVoice() {
	select {
	case c.voiceEv <- struct{}{}:
	default:
	}
}

// caption shows the utterance once its answer is certain to be heard,
// and the answer's text as the voice reaches it, a word at a time.
func (c *Cascade) caption(j job, text []byte) {
	if !c.userShown && j.audio != nil && c.played.Load() >= int64(resumeWindow)*int64(c.outRate)/int64(time.Second) {
		c.userShown = true
		c.obs.Heard(j.speaker, text, true)
	}
	if !c.userShown && j.audio != nil {
		return
	}
	n := wordEnd(c.said, min(len(c.said), c.voicedAt(c.played.Load())))
	if n > c.captioned {
		c.captioned = n
		c.obs.Said(c.said, n, false)
	}
}

// runCalls performs the tool calls of the last reply, once the turn that
// asked for them is over, adding their results to the conversation, and
// reports whether any ran: the agent then answers again knowing them. A
// reply dropped before its turn ends calls nothing.
func (c *Cascade) runCalls(ctx context.Context, j job) bool {
	calls := c.cfg.Session.Calls()
	if len(calls) == 0 || !c.awaitTurn(ctx, j.id) {
		return false
	}
	for _, call := range calls {
		result := chat.Run(ctx, c.cfg.Tools, call)
		c.obs.Stage(Called, time.Since(j.at))
		if err := c.cfg.Session.Add(chat.ToolResult, result); err != nil {
			c.fail(err)
			return false
		}
	}
	return true
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
	c.active()
	if err != nil {
		c.fail(err)
	}
}

func (c *Cascade) fail(err error) {
	c.obs.Error(err)
}

// errReplyEnded ends a reply at its silence marker: the model wrote what
// it had to.
var errReplyEnded = errors.New("duplex: reply ended")

// replyWriter receives the model's words as they are written. It holds
// back text that may be the start of a silence marker, or a repetition of
// the message being answered, speaks the rest through the voice, and ends
// the reply at a marker, keeping the time it asks for.
type replyWriter struct {
	c        *Cascade
	answers  []byte // the message being answered, for the echo guard
	spoken   int    // bytes of c.reply passed to the voice
	decided  bool   // the reply is known not to begin by repeating the message
	echoed   bool   // a repetition was cut
	trimLead bool   // ... and what follows it starts at its first word
	ended    bool   // a marker ended the reply
	wait     time.Duration
	until    string // what the marker named as ending the silence
	round    int    // bytes of c.reply before the last tool round's reply
	j        job
}

func (w *replyWriter) begin(j job, answers []byte) {
	w.answers, w.spoken, w.decided, w.echoed, w.trimLead, w.ended, w.wait, w.until, w.round, w.j = answers, 0, false, false, false, false, 0, "", 0, j
}

func (w *replyWriter) Write(p []byte) (int, error) {
	c := w.c
	if w.ended {
		return len(p), nil
	}
	if len(c.reply) == 0 {
		c.obs.Stage(FirstText, time.Since(w.j.at))
	}
	c.reply = append(c.reply, p...)
	for {
		unsent := c.reply[w.spoken:]
		lt := bytes.IndexByte(unsent, '<')
		if lt < 0 {
			return len(p), w.speak(len(unsent), false)
		}
		if lt > 0 {
			if err := w.speak(lt, false); err != nil {
				return len(p), err
			}
			unsent = c.reply[w.spoken:]
		}
		n, wait, until, ok := marker(unsent)
		switch {
		case n < 0:
			return len(p), nil // may be a marker: wait for more
		case !ok:
			// A '<' that is text.
			if err := w.speak(1, false); err != nil {
				return len(p), err
			}
			continue
		}
		w.ended, w.wait, w.until = true, wait, until
		return len(p), errReplyEnded
	}
}

// speak passes n unsent bytes of the reply to the voice, unless they may
// begin a repetition of the message; flush passes them whatever they are.
func (w *replyWriter) speak(n int, flush bool) error {
	c := w.c
	if n == 0 {
		return nil
	}
	if !w.decided {
		lead := c.reply[:w.spoken+n]
		switch k, holding := echoOf(view(lead), view(w.answers)); {
		case holding && !flush:
			return nil
		case k > 0:
			w.decided, w.echoed, w.trimLead = true, true, true
			w.spoken = k
			n = len(lead) - k
		default:
			w.decided = true
		}
	}
	src := c.reply[w.spoken : w.spoken+n]
	if w.trimLead {
		// After a cut repetition, its punctuation and spaces go too.
		rest := bytes.TrimLeftFunc(src, func(r rune) bool { return unicode.IsPunct(r) || unicode.IsSpace(r) })
		w.spoken += len(src) - len(rest)
		src = rest
		if len(src) == 0 {
			return nil
		}
		w.trimLead = false
	}
	c.said = appendSpeakable(c.said, src)
	w.spoken += len(src)
	c.spans = append(c.spans, span{len(c.said), w.spoken})
	c.mu.Lock()
	c.saying = append(c.saying[:0], c.said...)
	c.mu.Unlock()
	c.caption(w.j, w.answers)
	return c.voice()
}

// voice writes to the synthesizer what said holds beyond what it has, once
// said holds words; the first write begins the utterance, and the pump
// plays it. Until the voice first sounds, and for at most firstAudio, the
// writer then waits for it.
func (c *Cascade) voice() error {
	if c.voiceAt == len(c.said) || len(bytes.TrimSpace(c.said)) == 0 {
		return nil
	}
	ctx := c.replyCtx
	if !c.voicing {
		if err := c.cfg.Voice.Begin(ctx, c.cfg.Speak); err != nil {
			return err
		}
		c.voicing, c.firstSent = true, time.Now()
		c.pumpJobs <- c.rw.j
	}
	if _, err := c.cfg.Voice.Write(c.said[c.voiceAt:]); err != nil {
		return err
	}
	c.voiceAt = len(c.said)
	for !c.sounded.Load() && time.Since(c.firstSent) < firstAudio {
		select {
		case <-c.voiceEv:
		case <-c.tick.C:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// flush speaks what the writer still holds once the reply is written: a
// possible repetition that never became one, or a marker left unclosed.
func (w *replyWriter) flush() error {
	c := w.c
	if w.ended {
		return nil
	}
	unsent := c.reply[w.spoken:]
	if len(unsent) >= len(silentMark) && bytes.HasPrefix(unsent, []byte(silentMark)) {
		w.ended = true // "<silent" without its '>': silence all the same
		return nil
	}
	return w.speak(len(unsent), true)
}

// marker parses a silence marker at the start of b: <silent>, <silent
// 30s> with the time after which the agent wants to be asked again, or
// <silent until "hi"> with what ends the silence. It returns the marker's
// length, the time, the condition, and whether b starts with one; n is -1
// when b may be the beginning of a marker.
func marker(b []byte) (n int, wait time.Duration, until string, ok bool) {
	if !bytes.HasPrefix(b, []byte(silentMark)) {
		if bytes.HasPrefix([]byte(silentMark), b) {
			return -1, 0, "", false
		}
		return 0, 0, "", false
	}
	end := bytes.IndexByte(b, '>')
	if end < 0 {
		if len(b) > 64 {
			return 0, 0, "", false // too long to be one
		}
		return -1, 0, "", false
	}
	inner := b[len(silentMark):end]
	if _, after, found := bytes.Cut(inner, []byte("until")); found {
		// What is named, quoted or not; a bare "until" is being asked to
		// speak.
		until = string(bytes.Trim(bytes.TrimSpace(after), `"“”'`))
		if until == "" {
			until = "being asked to speak"
		}
		return end + 1, 0, until, true
	}
	// The number and unit, if any, wherever they are: "30s", "for 30
	// seconds", "2 minutes".
	i := bytes.IndexFunc(inner, func(r rune) bool { return r >= '0' && r <= '9' })
	if i < 0 {
		return end + 1, 0, "", true
	}
	var v int
	for i < len(inner) && inner[i] >= '0' && inner[i] <= '9' {
		v = v*10 + int(inner[i]-'0')
		i++
	}
	unit := bytes.TrimSpace(inner[i:])
	wait = time.Duration(v) * time.Second
	if len(unit) > 0 && unit[0]|0x20 == 'm' && !(len(unit) > 1 && unit[1]|0x20 == 's') {
		wait = time.Duration(v) * time.Minute
	}
	if wait > time.Hour {
		wait = time.Hour
	}
	return end + 1, wait, "", true
}

// echoOf reports how much of reply, from its start, repeats message word
// for word (0 if it does not), and whether reply, still being written, may
// yet turn out to; messages of fewer than three words are answered, not
// repeated.
func echoOf(reply, message string) (n int, holding bool) {
	said := strings.FieldsFunc(strings.ToLower(message), notWord)
	if len(said) < 3 {
		return 0, false
	}
	i, matched, open := 0, 0, false // open: the reply ends inside a word
	for matched < len(said) {
		for i < len(reply) {
			r, size := utf8.DecodeRuneInString(reply[i:])
			if !notWord(r) {
				break
			}
			i += size
		}
		if i == len(reply) {
			break
		}
		j := i
		for j < len(reply) {
			r, size := utf8.DecodeRuneInString(reply[j:])
			if notWord(r) {
				break
			}
			j += size
		}
		w := strings.ToLower(reply[i:j])
		if j == len(reply) {
			open = strings.HasPrefix(said[matched], w)
			break
		}
		if w != said[matched] {
			matched = -1
			break
		}
		matched++
		i = j
	}
	switch {
	case matched == len(said):
		return i, false
	case matched >= 0 && (open || i == len(reply)):
		holding = true
	}
	return 0, holding
}

func notWord(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '\'' }

// cut returns n backed off to a UTF-8 boundary of s.
func cut(s []byte, n int) int {
	if n >= len(s) {
		return len(s)
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return n
}

// wordEnd returns n backed off to the end of a whole word of s: captions
// show a word once the voice speaks it, and never half of one. Words of
// scripts written without spaces are single characters.
func wordEnd(s []byte, n int) int {
	n = cut(s, n)
	for n > 0 && n < len(s) {
		prev, size := utf8.DecodeLastRune(s[:n])
		next, _ := utf8.DecodeRune(s[n:])
		if !inWord(prev) || !inWord(next) {
			break
		}
		n -= size
	}
	return n
}

// inWord reports whether r continues a word written with spaces.
func inWord(r rune) bool {
	return (unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r) || r == '\'' || r == '’') &&
		!unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Thai)
}

// ring is a bounded FIFO of samples, safe for one writer and one reader.
// Each reset starts a generation, so that a writer can tell it is late.
type ring[T any] struct {
	mu   sync.Mutex
	buf  []T
	head int // next to read
	n    int
	gen  uint32
}

func newRing[T any](capacity int) *ring[T] { return &ring[T]{buf: make([]T, capacity)} }

// write appends v, dropping the oldest samples when full.
func (r *ring[T]) write(v []T) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.append(v)
}

// writeAt is write unless the ring was reset since generation gen.
func (r *ring[T]) writeAt(gen uint32, v []T) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.gen == gen {
		r.append(v)
	}
}

func (r *ring[T]) append(v []T) {
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
	r.gen++
	r.mu.Unlock()
}

// generation reports the ring's resets so far.
func (r *ring[T]) generation() uint32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.gen
}
