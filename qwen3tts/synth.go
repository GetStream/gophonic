// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/internal/whispergemm"
	"github.com/GetStream/gophonic/speech"
)

// Sampling defaults of the official generation config.
const (
	temperature   = 0.9
	topK          = 50
	repetition    = 1.05
	minFrames     = 2    // frames before the end token is allowed
	maxFrames     = 4096 // about 5.5 minutes of speech per utterance
	defaultVoice  = "ryan"
	placeholderID = 0 // the talker's and code predictor's ids are all rows
	seed          = 0x5eed
)

const (
	// audioFrames is the decoded speech a lane holds for Read: four
	// seconds, after which generation waits for the reader.
	audioFrames = 50
	// alignStep bounds how far the talker's alignment moves in one frame:
	// three text tokens in 80 ms is faster than anyone speaks.
	alignStep = 3
)

var (
	errIdle    = errors.New("qwen3tts: no utterance begun")
	errDropped = errors.New("qwen3tts: utterance dropped") // a later Begin superseded it
)

// Synthesizer is one lane of a Model: a speech.Synthesizer. Its worker
// goroutine speaks the current utterance: it tokenizes the text as it is
// written, generates each frame's codes on the GPU while another goroutine
// decodes the frame before, and queues the samples for Read with the text
// the talker was speaking in them.
type Synthesizer struct {
	m        *Model
	exec     *whispergemm.Executor
	tws, cws *qwen3lm.Workspace
	kv, ckv  *qwen3lm.PrefixKV

	hidden, cpHidden []float32
	logits, cpLogits []float32
	rows             []float32 // talker input rows
	row              []float32 // one talker input row
	cpRows           []float32 // code predictor input rows
	ids              []int     // placeholder ids for the rows
	frame            [groups]int
	seen             []bool // first-codebook codes generated this utterance
	seenIDs          []int
	sample           sampler
	draws            uint64 // the GPU's draws so far
	greedy           bool   // LaneOptions.greedy

	// The utterance's text as the worker has it: tokens, their projected
	// rows, the byte of the text each token ends at, and bytes not yet
	// tokenized.
	text      []int
	textRows  []float32
	textEnd   []int
	tokenized int // bytes of the text tokenized
	textAt    int // next text token to feed
	textDone  bool
	eosFed    bool
	pending   []byte
	tokWS     qwen3lm.TokenizerWorkspace
	tmp       []float32

	// The talker's alignment: its head's weights over a window of text
	// tokens from the one being spoken (at), which is at talker position
	// textStart+at.
	align     qwen3lm.Probe
	at        int
	textStart int

	// The codec decodes one frame on its own goroutine while the GPU
	// generates the next: jobs carries codes to it, done returns the
	// buffer it filled. The frame in flight is held with the text it
	// voices.
	dec      *decoder
	pcm      [2][]float32
	jobs     chan decodeJob
	done     chan int
	inFlight bool
	next     int   // the buffer the next frame decodes into
	flight   int32 // bytes of text voiced by the frame in flight
	first    bool  // the utterance's first frame is yet to be queued

	// The voice prompt: the rows before the first text token (the assistant
	// role and the codec's control tokens) depend only on the speaker and
	// language, and each utterance appends after them, so the talker cache
	// keeps them across utterances and a later utterance in the same voice
	// evaluates only the first text token's row.
	voice, voiceLang, voiceRows int // the cached prompt's speaker, language, and rows (0: none)
	voiceStyle                  string
	// The style instruction's tokens, "<|im_start|>user\n...<|im_end|>\n",
	// for the style they were made for.
	styleIDs []int
	styleFor string

	// The utterance, shared under mu by the caller's goroutines and the
	// worker. Begin counts utterances in id; the worker speaks utterance
	// cur and drops it once id moves on.
	mu                         sync.Mutex
	id, cur                    uint64
	ctx                        context.Context
	speaker, language          int
	style                      string
	written                    []byte    // the text, as written
	ended                      bool      // End was called
	taken                      int       // bytes of written the worker has
	over                       bool      // the worker is done: Read returns err after the audio
	err                        error     // what cut the utterance, if anything
	audio                      []float32 // decoded samples, a ring
	head, held                 int       // the ring's first unread sample, and samples held
	marks                      []int32   // per frame queued, the bytes of written it has voiced
	read                       int       // samples read
	closed                     bool
	begun, wrote, queued, room chan struct{} // wake-ups, each held once
	quit, stopped              chan struct{}

	onFeed func(row, hidden []float32) // tests: each talker input row and the state before it
}

var _ speech.Synthesizer = (*Synthesizer)(nil)

// LaneOptions configures a Synthesizer.
type LaneOptions struct {
	// Greedy takes each frame's likeliest codes instead of sampling them,
	// as the official model's reference outputs do: the same text gives
	// the same speech, for tests and fixtures.
	Greedy bool
}

// NewSynthesizer opens a lane over m.
func NewSynthesizer(m *Model, opts LaneOptions) (*Synthesizer, error) {
	if m == nil || m.talker == nil || m.cp == nil || m.codec == nil {
		return nil, fmt.Errorf("qwen3tts: nil or closed model")
	}
	s := &Synthesizer{m: m, greedy: opts.Greedy, hidden: make([]float32, m.hidden), cpHidden: make([]float32, m.cpHidden),
		logits: make([]float32, m.cfg.Talker.Vocab), cpLogits: make([]float32, codes),
		row: make([]float32, m.hidden), cpRows: make([]float32, 2*m.cpHidden), seen: make([]bool, m.cfg.Talker.Vocab),
		pcm:  [2][]float32{make([]float32, FrameSamples), make([]float32, FrameSamples)},
		jobs: make(chan decodeJob, 1), done: make(chan int, 1),
		audio: make([]float32, audioFrames*FrameSamples), marks: make([]int32, 0, maxFrames),
		begun: make(chan struct{}, 1), wrote: make(chan struct{}, 1), queued: make(chan struct{}, 1), room: make(chan struct{}, 1),
		quit: make(chan struct{}), stopped: make(chan struct{})}
	// A lane draws from one random stream across its utterances: the same
	// text varies from one to the next, as a speaker's voice does, and a
	// lane's utterances are reproducible.
	s.sample.seed(seed)
	if m.aligned {
		s.align = qwen3lm.Probe{Layer: m.align[0], Head: m.align[1], Probs: make([]float32, alignStep+1)}
	}
	var err error
	if s.exec, err = whispergemm.NewExecutor(m.threads); err != nil {
		return nil, err
	}
	if s.tws, err = m.tEval.NewWorkspace(m.threads); err != nil {
		return nil, err
	}
	if s.cws, err = m.cpEval.NewWorkspace(m.threads); err != nil {
		return nil, err
	}
	if s.ckv, err = m.cpEval.NewPrefixKV(groups + 1); err != nil {
		return nil, err
	}
	s.dec = m.codec.newDecoder()
	go s.decodeLoop()
	go s.work()
	return s, nil
}

type decodeJob struct {
	frame [groups]int
	buf   int
}

func (s *Synthesizer) decodeLoop() {
	defer s.dec.exec.Close()
	for job := range s.jobs {
		s.dec.decode(&job.frame, s.pcm[job.buf])
		s.done <- job.buf
	}
}

// wake holds one wake-up on c.
func wake(c chan struct{}) {
	select {
	case c <- struct{}{}:
	default:
	}
}

// SampleRate is 24 kHz.
func (s *Synthesizer) SampleRate() int { return SampleRate }

// Voices lists the preset speakers.
func (s *Synthesizer) Voices() []string { return s.m.voices }

// Begin starts an utterance; see speech.Synthesizer.
func (s *Synthesizer) Begin(ctx context.Context, opts speech.SpeakOptions) error {
	speaker, language, err := s.m.voiceOf(opts)
	if err != nil {
		return err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return speech.ErrClosed
	}
	s.id++
	s.ctx, s.speaker, s.language, s.style = ctx, speaker, language, opts.Style
	s.written, s.ended, s.taken, s.over, s.err = s.written[:0], false, 0, false, nil
	s.head, s.held, s.marks, s.read = 0, 0, s.marks[:0], 0
	s.mu.Unlock()
	// The worker may be waiting on the utterance this one drops.
	wake(s.begun)
	wake(s.wrote)
	wake(s.room)
	return nil
}

// Write adds text to the utterance; see speech.Synthesizer.
func (s *Synthesizer) Write(text []byte) (int, error) {
	s.mu.Lock()
	err := s.writable()
	if err == nil {
		s.written = append(s.written, text...)
	}
	s.mu.Unlock()
	if err != nil {
		return 0, err
	}
	wake(s.wrote)
	return len(text), nil
}

// End marks the utterance's text complete.
func (s *Synthesizer) End() error {
	s.mu.Lock()
	err := s.writable()
	s.ended = true
	s.mu.Unlock()
	wake(s.wrote)
	return err
}

// writable reports why the utterance takes no more text, if it does not.
func (s *Synthesizer) writable() error {
	switch {
	case s.closed:
		return speech.ErrClosed
	case s.id == 0 || s.ended:
		return errIdle
	case s.over && s.err != nil:
		return s.err
	}
	return s.ctx.Err()
}

// Read pulls the utterance's samples; see speech.Synthesizer.
func (s *Synthesizer) Read(pcm []float32) (int, error) {
	for {
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			return 0, speech.ErrClosed
		}
		if s.id == 0 {
			s.mu.Unlock()
			return 0, errIdle
		}
		ctx := s.ctx
		if err := ctx.Err(); err != nil {
			s.mu.Unlock()
			return 0, err
		}
		if len(pcm) == 0 {
			s.mu.Unlock()
			return 0, nil
		}
		if s.held > 0 {
			n := min(len(pcm), s.held)
			k := copy(pcm[:n], s.audio[s.head:])
			copy(pcm[k:n], s.audio)
			s.head, s.held, s.read = (s.head+n)%len(s.audio), s.held-n, s.read+n
			s.mu.Unlock()
			wake(s.room)
			return n, nil
		}
		if s.over {
			err := s.err
			if err == nil {
				err = io.EOF
			}
			s.mu.Unlock()
			return 0, err
		}
		s.mu.Unlock()
		select {
		case <-s.queued:
		case <-ctx.Done():
		case <-s.quit:
		}
	}
}

// Voiced reports the bytes of the utterance's text its first samples
// samples speak: the text up to the token the talker's alignment head
// attended to in the frame that holds the last of them. A checkpoint
// without a known head reports the text spoken only at the end.
func (s *Synthesizer) Voiced(samples int) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	decoded := s.read + s.held
	switch {
	case samples <= 0:
		return 0
	case s.over && s.err == nil && samples >= decoded:
		return len(s.written)
	case len(s.marks) == 0:
		return 0
	}
	return int(s.marks[(min(samples, decoded)-1)/FrameSamples])
}

// Close releases the lane, once its worker is done with the frame it
// generates.
func (s *Synthesizer) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.mu.Unlock()
	close(s.quit)
	<-s.stopped
	s.exec.Close()
	close(s.jobs)
	return errors.Join(s.tws.Close(), s.cws.Close())
}

// work speaks each utterance Begin starts, until the lane closes.
func (s *Synthesizer) work() {
	defer close(s.stopped)
	for {
		select {
		case <-s.begun:
		case <-s.quit:
			return
		}
		s.mu.Lock()
		s.cur = s.id
		ctx, speaker, language, style := s.ctx, s.speaker, s.language, s.style
		s.mu.Unlock()
		err := s.speak(ctx, speaker, language, style)
		s.mu.Lock()
		if s.id == s.cur {
			s.over, s.err = true, err
		}
		s.mu.Unlock()
		wake(s.queued)
	}
}

// speak generates the current utterance and queues its samples for Read.
// Each frame's codes go to the decoder goroutine while the frame before is
// queued, so decoding overlaps generation; the first frame, which the
// listener waits for, is queued as soon as it is decoded.
func (s *Synthesizer) speak(ctx context.Context, speaker, language int, style string) error {
	s.first = true
	err := s.generate(ctx, speaker, language, style, s.emit)
	if err == nil {
		err = s.flush()
	}
	if s.inFlight {
		<-s.done
		s.inFlight = false
	}
	if errors.Is(err, io.EOF) {
		err = nil
	}
	return err
}

// emit sends a frame's codes to the decoder and queues the frame before.
func (s *Synthesizer) emit(frame *[groups]int) error {
	if err := s.flush(); err != nil {
		return err
	}
	s.jobs <- decodeJob{*frame, s.next}
	s.inFlight, s.flight, s.next = true, int32(s.voicedAt()), 1-s.next
	if s.first {
		s.first = false
		return s.flush()
	}
	return nil
}

// flush queues the frame in flight once it is decoded.
func (s *Synthesizer) flush() error {
	if !s.inFlight {
		return nil
	}
	buf := <-s.done
	s.inFlight = false
	return s.queue(s.pcm[buf], s.flight)
}

// queue adds a frame's samples to the ring with the text they voiced,
// waiting for the reader to make room.
func (s *Synthesizer) queue(pcm []float32, voiced int32) error {
	s.mu.Lock()
	for {
		if s.id != s.cur {
			s.mu.Unlock()
			return errDropped
		}
		if s.closed {
			s.mu.Unlock()
			return speech.ErrClosed
		}
		if err := s.ctx.Err(); err != nil {
			s.mu.Unlock()
			return err
		}
		if len(s.audio)-s.held >= len(pcm) {
			break
		}
		ctx := s.ctx
		s.mu.Unlock()
		select {
		case <-s.room:
		case <-ctx.Done():
		case <-s.quit:
		}
		s.mu.Lock()
	}
	at := (s.head + s.held) % len(s.audio)
	k := copy(s.audio[at:], pcm)
	copy(s.audio, pcm[k:])
	s.held += len(pcm)
	s.marks = append(s.marks, voiced)
	s.mu.Unlock()
	wake(s.queued)
	return nil
}

// take moves the text written since the last take to pending, waiting for
// some; it returns io.EOF once the text has ended and all of it is taken.
func (s *Synthesizer) take(ctx context.Context) error {
	s.mu.Lock()
	for {
		switch {
		case s.id != s.cur:
			s.mu.Unlock()
			return errDropped
		case s.closed:
			s.mu.Unlock()
			return speech.ErrClosed
		case s.taken < len(s.written):
			s.pending = append(s.pending, s.written[s.taken:]...)
			s.taken = len(s.written)
			s.mu.Unlock()
			return nil
		case s.ended:
			s.mu.Unlock()
			return io.EOF
		}
		s.mu.Unlock()
		select {
		case <-s.wrote:
		case <-ctx.Done():
			return ctx.Err()
		case <-s.quit:
		}
		s.mu.Lock()
	}
}

// voicedAt is the text voiced by the frame generated last: up to the end of
// the token the talker attends to, or none without an alignment head (Read
// reports all of it at the end).
func (s *Synthesizer) voicedAt() int {
	if !s.m.aligned || len(s.textEnd) == 0 {
		return 0
	}
	return s.textEnd[min(s.at, len(s.textEnd)-1)]
}

// generate runs the talker and code predictor over the utterance's text,
// passing each frame's codes to emit.
func (s *Synthesizer) generate(ctx context.Context, speaker, language int, style string, emit func(*[groups]int) error) error {
	c := &s.m.cfg.Talker
	s.reset()
	s.dec.reset()
	// The prompt needs the first text token.
	for len(s.text) == 0 && !s.textDone {
		if err := s.pull(ctx); err != nil {
			return err
		}
	}
	if len(s.text) == 0 {
		return nil // no text
	}
	if err := s.prefill(speaker, language, style); err != nil {
		return err
	}
	for step := 0; step < maxFrames; step++ {
		if err := s.talkerLogits(); err != nil {
			return err
		}
		c0 := s.pick(s.logits, step)
		if c0 == c.CodecEOS {
			return nil
		}
		s.frame[0] = c0
		if !s.seen[c0] {
			s.seen[c0] = true
			s.seenIDs = append(s.seenIDs, c0)
		}
		if err := s.predict(); err != nil {
			return err
		}
		if err := emit(&s.frame); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := s.feed(ctx); err != nil {
			return err
		}
	}
	return nil
}

// voiceOf finds the talker's speaker and language for opts.
func (m *Model) voiceOf(opts speech.SpeakOptions) (speaker, language int, err error) {
	c := &m.cfg.Talker
	voice := opts.Voice
	if voice == "" {
		voice = defaultVoice
	}
	speaker, ok := lookup(c.Speakers, voice)
	if !ok {
		return 0, 0, fmt.Errorf("qwen3tts: unknown voice %q: %w", opts.Voice, speech.ErrUnsupported)
	}
	language = -1
	if opts.Language != speech.Unknown {
		id, known := lookup(c.Languages, opts.Language.Name())
		if !known {
			return 0, 0, fmt.Errorf("qwen3tts: cannot speak %v: %w", opts.Language, speech.ErrUnsupported)
		}
		language = id
	}
	if dialect, ok := lookup(c.Dialects, voice); ok && (language < 0 || language == c.Languages["chinese"]) {
		if name, ok := dialect.(string); ok {
			language = c.Languages[name]
		}
	}
	return speaker, language, nil
}

// lookup finds name in m regardless of case, without allocating.
func lookup[V any](m map[string]V, name string) (V, bool) {
	for k, v := range m {
		if strings.EqualFold(k, name) {
			return v, true
		}
	}
	var zero V
	return zero, false
}

// reset clears the utterance state.
func (s *Synthesizer) reset() {
	for _, id := range s.seenIDs {
		s.seen[id] = false
	}
	s.seenIDs = s.seenIDs[:0]
	s.text, s.textRows, s.textEnd, s.pending = s.text[:0], s.textRows[:0], s.textEnd[:0], s.pending[:0]
	s.tokenized, s.textAt, s.textDone, s.eosFed, s.at = 0, 0, false, false, 0
}

// pull takes the text written since the last pull and tokenizes what ends
// at a word boundary. Until the prompt has its first token, a first word
// with nothing after it yet is tokenized as it stands, so that speech can
// start: only that word's tokens may differ from the whole text's.
func (s *Synthesizer) pull(ctx context.Context) error {
	err := s.take(ctx)
	if errors.Is(err, io.EOF) {
		s.textDone = true
		return s.tokenize(len(s.pending))
	}
	if err != nil {
		return err
	}
	// Tokenize up to the last white space: Qwen's pre-tokenizer starts a
	// word with its leading space, so the split changes no token.
	cut := lastSpace(s.pending)
	if len(s.text) == 0 && cut <= 0 {
		cut = trimSpace(s.pending)
	}
	if cut <= 0 {
		return nil
	}
	return s.tokenize(cut)
}

func lastSpace(b []byte) int {
	for i := len(b); i > 0; {
		r, size := utf8.DecodeLastRune(b[:i])
		if unicode.IsSpace(r) {
			return i - size
		}
		i -= size
	}
	return -1
}

// trimSpace returns the length of b without trailing white space.
func trimSpace(b []byte) int {
	for i := len(b); i > 0; {
		r, size := utf8.DecodeLastRune(b[:i])
		if !unicode.IsSpace(r) {
			return i
		}
		i -= size
	}
	return 0
}

// tokenize appends the tokens of s.pending[:n], the byte of the text each
// ends at, and their projected rows.
func (s *Synthesizer) tokenize(n int) error {
	if n == 0 {
		return nil
	}
	start := len(s.text)
	s.text = slices.Grow(s.text, n+1)
	got, err := s.m.tokens.EncodeInto(stringOf(s.pending[:n]), s.text[start:start:cap(s.text)], &s.tokWS)
	if err != nil {
		return fmt.Errorf("qwen3tts: tokenize: %w", err)
	}
	s.text = s.text[:start+len(got)]
	// Qwen's byte-level tokens spell the text: their bytes add up to it.
	end := s.tokenized
	for _, id := range got {
		end += len(s.m.tokens.Piece(id))
		s.textEnd = append(s.textEnd, min(end, s.tokenized+n))
	}
	s.tokenized += n
	if len(got) > 0 {
		s.textEnd[len(s.textEnd)-1] = s.tokenized
	}
	s.pending = s.pending[:copy(s.pending, s.pending[n:])]
	h := s.m.hidden
	added := len(s.text) - start
	s.textRows = slices.Grow(s.textRows, added*h)[:len(s.text)*h]
	s.tmp = slices.Grow(s.tmp[:0], added*h)[:added*h]
	return s.m.textRows(s.exec, s.textRows[start*h:], s.text[start:], s.tmp)
}

// prefill evaluates the prompt: the assistant role, the codec's control
// tokens (language, speaker) under TTS padding, and the first text token.
// With the voice prompt cached, only the text token's row.
func (s *Synthesizer) prefill(speaker, language int, style string) error {
	m, c, h := s.m, &s.m.cfg.Talker, s.m.hidden
	var ids [7]int
	control := append(ids[:0], c.NoThink, c.ThinkBOS, c.ThinkEOS)
	if language >= 0 {
		control = append(ids[:0], c.Think, c.ThinkBOS, language, c.ThinkEOS)
	}
	control = append(control, speaker, c.CodecPad, c.CodecBOS)
	// A style is an instruction the talker reads before everything else,
	// as the official generate_custom_voice's instruct.
	if style != s.styleFor {
		s.styleIDs, s.styleFor = s.styleIDs[:0], style
		if style != "" {
			text := "<|im_start|>user\n" + style + "<|im_end|>\n"
			got, err := m.tokens.EncodeInto(text, make([]int, 0, len(text)+8), &s.tokWS)
			if err != nil {
				s.styleFor = ""
				return fmt.Errorf("qwen3tts: tokenize style: %w", err)
			}
			s.styleIDs = got
		}
	}
	ns := len(s.styleIDs)
	n := ns + 3 + len(control)
	s.textAt = 1
	if s.voiceRows == n-1 && s.voice == speaker && s.voiceLang == language && s.voiceStyle == style &&
		s.kv != nil && len(s.kv.Tokens()) >= n-1 {
		s.rows = slices.Grow(s.rows[:0], h)[:h]
		copy(s.rows, s.textRows[:h])
		m.addCodec(s.rows, c.CodecBOS)
		s.textStart = n - 1
		return s.talkerRows(1, n-1, false)
	}
	s.voiceRows = 0
	s.rows = slices.Grow(s.rows[:0], n*h)[:n*h]
	s.tmp = slices.Grow(s.tmp[:0], (ns+3)*h)[:(ns+3)*h]
	if ns > 0 {
		if err := m.textRows(s.exec, s.rows[:ns*h], s.styleIDs, s.tmp); err != nil {
			return err
		}
	}
	// "<|im_start|>assistant\n", whose ids the tokenizer shares with Qwen3.
	if err := m.textRows(s.exec, s.rows[ns*h:(ns+3)*h], roleIDs[:], s.tmp); err != nil {
		return err
	}
	for i, id := range control[:len(control)-1] {
		row := s.rows[(ns+3+i)*h : (ns+4+i)*h]
		text := m.padRow
		if i == len(control)-2 {
			text = m.bosRow
		}
		copy(row, text)
		m.addCodec(row, id)
	}
	last := s.rows[(n-1)*h:]
	copy(last, s.textRows[:h])
	m.addCodec(last, c.CodecBOS)
	if err := s.talkerRows(n, 0, false); err != nil {
		return err
	}
	s.voice, s.voiceLang, s.voiceStyle, s.voiceRows = speaker, language, style, n-1
	s.textStart = n - 1
	return nil
}

// roleIDs is "<|im_start|>assistant\n" in the Qwen tokenizer.
var roleIDs = [3]int{151644, 77091, 198}

// addCodec adds the talker's codec embedding of id to row.
func (m *Model) addCodec(row []float32, id int) {
	addBF16(row, m.codecEmbed[id*m.hidden:(id+1)*m.hidden])
}

func addBF16(dst []float32, src []uint16) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] += q8gemm.BF16ToF32(src[i])
	}
}

// talkerRows evaluates n rows of s.rows after keep stored positions; with
// align, the one row's alignment head moves s.at to the text token it
// attends to most, within alignStep of the last.
func (s *Synthesizer) talkerRows(n, keep int, align bool) error {
	if s.kv == nil || s.kv.Capacity() < keep+n+1 {
		capacity := 1024
		for capacity < keep+n+maxFrames/4 {
			capacity *= 2
		}
		kv, err := s.m.tEval.NewPrefixKV(min(capacity, s.m.cfg.Talker.MaxPositions))
		if err != nil {
			return err
		}
		if s.kv != nil && keep > 0 {
			kv.CopyPrefix(s.kv, keep)
		}
		s.kv = kv
	}
	if keep+n > s.kv.Capacity() {
		return errors.New("qwen3tts: utterance too long")
	}
	s.ids = placeholders(s.ids, n)
	embeds := qwen3lm.Embeds{Token: placeholderID, Rows: s.rows[:n*s.m.hidden]}
	if !align {
		return s.m.tEval.HiddenLastExtendEmbedInto(s.kv, keep, s.ids, embeds, s.hidden, s.tws)
	}
	s.align.From = s.textStart + s.at
	if err := s.m.tEval.HiddenLastExtendProbeInto(s.kv, keep, s.ids, embeds, s.hidden, &s.align, s.tws); err != nil {
		return err
	}
	best := 0
	for i, p := range s.align.Probs {
		if p > s.align.Probs[best] {
			best = i
		}
	}
	s.at += best
	return nil
}

func placeholders(ids []int, n int) []int {
	ids = slices.Grow(ids[:0], n)[:n]
	for i := range ids {
		ids[i] = placeholderID
	}
	return ids
}

func (s *Synthesizer) talkerLogits() error {
	return s.m.head.Mul(s.logits, s.hidden)
}

// predict fills the frame's codebooks 1–15 from the talker state and the
// first codebook.
func (s *Synthesizer) predict() error {
	m, ch := s.m, s.m.cpHidden
	if err := m.proj.apply(s.exec, s.cpRows[:ch], s.hidden, 1); err != nil {
		return err
	}
	copy(s.cpRows[ch:2*ch], m.cpRows[0][s.frame[0]*ch:(s.frame[0]+1)*ch])
	s.ids = placeholders(s.ids, 2)
	if m.cpDecode != nil {
		// The fifteen codes in one GPU submission.
		d := qwen3lm.Sampling{TopK: topK, Temperature: temperature, Seed: seed, Draw: s.draws}
		if s.greedy {
			d.TopK = 1
		}
		s.draws += groups - 1
		return m.cpEval.DecodeInto(m.cpDecode, s.ckv, 0, s.ids, qwen3lm.Embeds{Token: placeholderID, Rows: s.cpRows}, d, s.frame[1:], nil, s.cws)
	}
	if err := m.cpEval.HiddenLastExtendEmbedInto(s.ckv, 0, s.ids, qwen3lm.Embeds{Token: placeholderID, Rows: s.cpRows}, s.cpHidden, s.cws); err != nil {
		return err
	}
	for g := 1; g < groups; g++ {
		if err := m.heads[g-1].Mul(s.cpLogits, s.cpHidden); err != nil {
			return err
		}
		s.frame[g] = s.pickPredictor(s.cpLogits)
		if g == groups-1 {
			break
		}
		code := s.frame[g]
		s.ids = placeholders(s.ids, 1)
		rows := m.cpRows[g][code*ch : (code+1)*ch]
		if err := m.cpEval.HiddenLastExtendEmbedInto(s.ckv, g+1, s.ids, qwen3lm.Embeds{Token: placeholderID, Rows: rows}, s.cpHidden, s.cws); err != nil {
			return err
		}
	}
	return nil
}

// feed evaluates the talker's next input: the frame's codec embeddings
// plus the next text token's row (the TTS end once the text is spoken, and
// padding after it).
func (s *Synthesizer) feed(ctx context.Context) error {
	m, h := s.m, s.m.hidden
	for s.textAt >= len(s.text) && !s.textDone {
		if err := s.pull(ctx); err != nil {
			return err
		}
	}
	row := s.row
	switch {
	case s.textAt < len(s.text):
		copy(row, s.textRows[s.textAt*h:(s.textAt+1)*h])
		s.textAt++
	case !s.eosFed:
		copy(row, m.eosRow)
		s.eosFed = true
	default:
		copy(row, m.padRow)
	}
	m.addCodec(row, s.frame[0])
	for g := 1; g < groups; g++ {
		code := s.frame[g]
		addBF16(row, m.cpEmbed[g-1][code*h:(code+1)*h])
	}
	if s.onFeed != nil {
		s.onFeed(row, s.hidden)
	}
	s.rows = append(s.rows[:0], row...)
	return s.talkerRows(1, len(s.kv.Tokens()), m.aligned)
}

// pick draws the frame's first codebook as the official generation does:
// repetition penalty over this utterance's codes, no end before minFrames,
// control tokens suppressed, then temperature and top-k sampling.
func (s *Synthesizer) pick(logits []float32, step int) int {
	c := &s.m.cfg.Talker
	for _, id := range s.seenIDs {
		if logits[id] < 0 {
			logits[id] *= repetition
		} else {
			logits[id] /= repetition
		}
	}
	inf := float32(math.Inf(-1))
	if step < minFrames {
		logits[c.CodecEOS] = inf
	}
	for i := c.Vocab - 1024; i < c.Vocab; i++ {
		if i != c.CodecEOS {
			logits[i] = inf
		}
	}
	if s.greedy {
		return argmax(logits)
	}
	return s.sample.topK(logits, topK, temperature)
}

func (s *Synthesizer) pickPredictor(logits []float32) int {
	if s.greedy {
		return argmax(logits)
	}
	return s.sample.topK(logits, topK, temperature)
}

func argmax(v []float32) int {
	best, at := v[0], 0
	for i, x := range v {
		if x > best {
			best, at = x, i
		}
	}
	return at
}
