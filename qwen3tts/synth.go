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
	maxFrames     = 4096 // about 5.5 minutes of speech per call
	defaultVoice  = "ryan"
	placeholderID = 0 // the talker's and code predictor's ids are all rows
)

// Synthesizer is one lane of a Model: a speech.Synthesizer.
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
	// Greedy decodes deterministically, as the reference tests do.
	Greedy bool

	// Text of the utterance: tokens, their projected rows, and bytes not
	// yet tokenized.
	text     []int
	textRows []float32
	textAt   int // next text token to feed
	textDone bool
	eosFed   bool
	pending  []byte
	tokWS    qwen3lm.TokenizerWorkspace
	tmp      []float32

	// The codec decodes one frame on its own goroutine while the GPU
	// generates the next: jobs carries codes to it, done returns the
	// buffer it filled.
	dec  *decoder
	pcm  [2][]float32
	jobs chan decodeJob
	done chan int

	onFeed func(row, hidden []float32) // tests: each talker input row and the state before it
}

var _ speech.Synthesizer = (*Synthesizer)(nil)

// NewSynthesizer opens a lane.
func NewSynthesizer(m *Model) (*Synthesizer, error) {
	s := &Synthesizer{m: m, hidden: make([]float32, m.hidden), cpHidden: make([]float32, m.cpHidden),
		logits: make([]float32, m.cfg.Talker.Vocab), cpLogits: make([]float32, codes),
		row: make([]float32, m.hidden), cpRows: make([]float32, 2*m.cpHidden), seen: make([]bool, m.cfg.Talker.Vocab),
		pcm:  [2][]float32{make([]float32, FrameSamples), make([]float32, FrameSamples)},
		jobs: make(chan decodeJob, 1), done: make(chan int, 1)}
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
	return s, nil
}

type decodeJob struct {
	frame [groups]int
	buf   int
}

func (s *Synthesizer) decodeLoop() {
	for job := range s.jobs {
		s.dec.decode(&job.frame, s.pcm[job.buf])
		s.done <- job.buf
	}
}

// SampleRate is 24 kHz.
func (s *Synthesizer) SampleRate() int { return SampleRate }

// Voices lists the preset speakers.
func (s *Synthesizer) Voices() []string { return s.m.Speakers() }

// Close releases the lane.
func (s *Synthesizer) Close() error {
	if s.exec == nil {
		return nil
	}
	s.exec.Close()
	s.exec = nil
	close(s.jobs)
	return errors.Join(s.tws.Close(), s.cws.Close())
}

// Speak speaks text as it arrives; see speech.Synthesizer.
// Each frame's codes go to the codec goroutine; the previous frame's
// samples are passed to out meanwhile, so decoding overlaps generation.
func (s *Synthesizer) Speak(ctx context.Context, opts speech.SpeakOptions, next func() ([]byte, error), out func([]float32) error) error {
	inFlight, buf := false, 0
	var outErr error
	flush := func() {
		if inFlight {
			filled := <-s.done
			inFlight = false
			if outErr == nil {
				outErr = out(s.pcm[filled])
			}
		}
	}
	err := s.generate(ctx, opts, next, func(frame *[groups]int) error {
		flush()
		if outErr != nil {
			return outErr
		}
		s.jobs <- decodeJob{*frame, buf}
		inFlight, buf = true, 1-buf
		return nil
	})
	flush()
	if err == nil {
		err = outErr
	}
	return err
}

// generate runs the talker and code predictor, passing each frame's codes
// to emit.
func (s *Synthesizer) generate(ctx context.Context, opts speech.SpeakOptions, next func() ([]byte, error), emit func(*[groups]int) error) error {
	if s.exec == nil {
		return speech.ErrClosed
	}
	c := &s.m.cfg.Talker
	voice := strings.ToLower(opts.Voice)
	if voice == "" {
		voice = defaultVoice
	}
	speaker, ok := c.Speakers[voice]
	if !ok {
		return fmt.Errorf("qwen3tts: unknown voice %q: %w", opts.Voice, speech.ErrUnsupported)
	}
	language := -1
	if opts.Language != "" {
		name, ok := speech.LanguageName(opts.Language)
		id, known := c.Languages[strings.ToLower(name)]
		if !ok || !known {
			return fmt.Errorf("qwen3tts: cannot speak %q: %w", opts.Language, speech.ErrUnsupported)
		}
		language = id
	}
	if dialect, ok := c.Dialects[voice].(string); ok && (language < 0 || language == c.Languages["chinese"]) {
		language = c.Languages[dialect]
	}
	s.reset()
	s.dec.reset()
	// The prompt needs the first text token.
	for len(s.text) == 0 && !s.textDone {
		if err := s.pull(next); err != nil {
			return err
		}
	}
	if len(s.text) == 0 {
		return nil // no text
	}
	if err := s.prefill(speaker, language); err != nil {
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
		if err := s.feed(next); err != nil {
			return err
		}
	}
	return nil
}

// reset clears the utterance state.
func (s *Synthesizer) reset() {
	for _, id := range s.seenIDs {
		s.seen[id] = false
	}
	s.seenIDs = s.seenIDs[:0]
	s.text, s.textRows, s.pending = s.text[:0], s.textRows[:0], s.pending[:0]
	s.textAt, s.textDone, s.eosFed = 0, false, false
	s.sample.seed(0x5eed)
}

// pull reads one piece of text and tokenizes what ends at a word boundary.
func (s *Synthesizer) pull(next func() ([]byte, error)) error {
	piece, err := next()
	s.pending = append(s.pending, piece...)
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

// tokenize appends the tokens of s.pending[:n] and their projected rows.
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
	s.pending = s.pending[:copy(s.pending, s.pending[n:])]
	h := s.m.hidden
	added := len(s.text) - start
	s.textRows = slices.Grow(s.textRows, added*h)[:len(s.text)*h]
	s.tmp = slices.Grow(s.tmp[:0], added*h)[:added*h]
	return s.m.textRows(s.exec, s.textRows[start*h:], s.text[start:], s.tmp)
}

// prefill evaluates the prompt: the assistant role, the codec's control
// tokens (language, speaker) under TTS padding, and the first text token.
func (s *Synthesizer) prefill(speaker, language int) error {
	m, c, h := s.m, &s.m.cfg.Talker, s.m.hidden
	control := []int{c.NoThink, c.ThinkBOS, c.ThinkEOS}
	if language >= 0 {
		control = []int{c.Think, c.ThinkBOS, language, c.ThinkEOS}
	}
	control = append(control, speaker, c.CodecPad, c.CodecBOS)
	n := 3 + len(control)
	s.rows = slices.Grow(s.rows[:0], n*h)[:n*h]
	s.tmp = slices.Grow(s.tmp[:0], 3*h)[:3*h]
	// "<|im_start|>assistant\n", whose ids the tokenizer shares with Qwen3.
	if err := m.textRows(s.exec, s.rows[:3*h], roleIDs[:], s.tmp); err != nil {
		return err
	}
	for i, id := range control[:len(control)-1] {
		row := s.rows[(3+i)*h : (4+i)*h]
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
	s.textAt = 1
	return s.talkerRows(n, 0)
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

// talkerRows evaluates n rows of s.rows after keep stored positions.
func (s *Synthesizer) talkerRows(n, keep int) error {
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
	return s.m.tEval.HiddenLastExtendEmbedInto(s.kv, keep, s.ids, qwen3lm.Embeds{Token: placeholderID, Rows: s.rows[:n*s.m.hidden]}, s.hidden, s.tws)
}

func placeholders(ids []int, n int) []int {
	ids = slices.Grow(ids[:0], n)[:n]
	for i := range ids {
		ids[i] = placeholderID
	}
	return ids
}

func (s *Synthesizer) talkerLogits() error {
	return s.m.tEval.LogitsInto(s.hidden, s.logits, s.tws)
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
func (s *Synthesizer) feed(next func() ([]byte, error)) error {
	m, h := s.m, s.m.hidden
	for s.textAt >= len(s.text) && !s.textDone {
		if err := s.pull(next); err != nil {
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
	return s.talkerRows(1, len(s.kv.Tokens()))
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
	if s.Greedy {
		return argmax(logits)
	}
	return s.sample.topK(logits, topK, temperature)
}

func (s *Synthesizer) pickPredictor(logits []float32) int {
	if s.Greedy {
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
