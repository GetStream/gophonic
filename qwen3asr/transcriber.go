// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"unsafe"

	"github.com/GetStream/gophonic/internal/mel"
	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

const (
	sampleRate = speech.SampleRate
	// maxChunkSamples is the longest audio transcribed in one pass; longer
	// audio is cut near quiet points, as the reference does at 1200 s.
	maxChunkSamples = 1200 * sampleRate
	// minChunkSamples is the length a cut piece is padded to.
	minChunkSamples = sampleRate / 2
	// minNewTokens bounds the generated tokens of short audio; longer audio
	// may generate newTokensPerSecond per second of speech.
	minNewTokens       = 512
	newTokensPerSecond = 16
	// maxDraft bounds the tokens of a partial transcript checked in one
	// pass, and verifyRows the draft positions whose logits are computed
	// at a time.
	maxDraft   = 192
	verifyRows = 32
)

var _ speech.Transcriber = (*Transcriber)(nil)

// Transcriber is one Qwen3-ASR call lane: the feature frontend, encoder and
// decoder workers, key-value cache, and every buffer a transcription needs.
// Calls must not overlap; open one Transcriber per concurrent lane. Numeric
// scratch and packed KV storage are reused after warming. The Go runtime may
// still allocate worker scheduling metadata.
type Transcriber struct {
	m               *Model
	frontend        *mel.Spectrogram
	enc             *encoderWorkspace    // CPU encoder
	encPrefix       encoderPrefix        // exact completed-window reuse for growing audio
	reusedAudioRows int                  // validated embedding prefix from the current encode
	audioRevision   uint64               // advances even when an encode or later prompt step fails
	decoderPrefix   decoderPrefix        // provenance of audio rows in the lane's existing KV cache
	genc            *gpuEncoderWorkspace // GPU encoder, when the model has one
	lm              *qwen3lm.Workspace
	kv              *qwen3lm.PrefixKV
	tokWS           qwen3lm.TokenizerWorkspace

	pcm      []float32 // normalized or padded copy of the input, when needed
	features []float32
	embeds   []float32
	prompt   []byte
	ids      []int
	hidden   []float32
	logits   []float32
	gen      []int
	draft    []int // a partial transcript's tokens
	// What the languages given limit (nil: nothing): the language names
	// the output may give, and the tokens the transcript may contain.
	limits     *limits
	limit      bool      // this pass names only an allowed language
	forcedText bool      // the prompt names the language: every token is text
	tail       []float32 // the states that check the draft
	vlogits    []float32 // their logits, verifyRows at a time
	raw        []byte
	runes      []rune
	fixed      []rune
	text       []byte
	closed     bool
}

// LaneOptions configures a Transcriber.
type LaneOptions struct {
	// Threads bounds the lane's CPU workers, including the caller; zero
	// picks GOMAXPROCS, at most 16, and at most 8 for the encoder.
	Threads int
}

// NewTranscriber opens a lane over m.
func NewTranscriber(m *Model, opts LaneOptions) (*Transcriber, error) {
	if m == nil {
		return nil, errors.New("qwen3asr: nil model")
	}
	workers := opts.Threads
	if workers < 0 || workers > 64 {
		return nil, fmt.Errorf("qwen3asr: invalid thread count %d", workers)
	}
	if workers == 0 {
		workers = min(runtime.GOMAXPROCS(0), 16)
	}
	t := &Transcriber{m: m, frontend: mel.NewSpectrogram(m.enc.freq[0], 0)}
	if m.genc != nil {
		t.genc = m.genc.newWorkspace()
	} else {
		enc, err := newEncoderWorkspace(m.enc, min(workers, 8))
		if err != nil {
			return nil, err
		}
		t.enc = enc
	}
	lm, err := m.eval.NewWorkspace(workers)
	if err != nil {
		t.closeEncoder()
		return nil, err
	}
	c := m.lm.Config()
	t.lm, t.hidden, t.logits = lm, make([]float32, c.Hidden), make([]float32, c.Vocab)
	return t, nil
}

func (t *Transcriber) closeEncoder() {
	t.encPrefix.close()
	if t.enc != nil {
		t.enc.close()
	}
	if t.genc != nil {
		t.genc.close()
	}
}

// Close releases the lane's workers. It is safe to call more than once.
func (t *Transcriber) Close() error {
	if t == nil || t.closed {
		return nil
	}
	t.closed = true
	t.closeEncoder()
	err := t.lm.Close()
	if e := t.kv.Close(); err == nil {
		err = e
	}
	t.kv = nil
	return err
}

// Transcribe implements speech.Transcriber. Options.Language forces the
// language, which skips detection; Options.Context primes recognition with
// names and terms. Audio longer than 20 minutes is transcribed in pieces cut
// at quiet points; the transcript reports the first piece's language.
// Qwen3-ASR produces no timestamps, so a Segments request yields one segment
// per piece, and Words is unsupported.
func (t *Transcriber) Transcribe(ctx context.Context, pcm []float32, opts speech.Options, dst *speech.Transcript) error {
	if t == nil || t.closed {
		return speech.ErrClosed
	}
	if opts.Words {
		return fmt.Errorf("qwen3asr: word timing: %w", speech.ErrUnsupported)
	}
	if opts.Turn && t.m.turn == nil {
		return fmt.Errorf("qwen3asr: turn judgment for this model: %w", speech.ErrUnsupported)
	}
	language := opts.Language
	if language == speech.Unknown && opts.Languages.Len() == 1 {
		for l := range opts.Languages.All() {
			language = l
		}
	}
	if language != speech.Unknown && !t.m.languages.Has(language) {
		return fmt.Errorf("qwen3asr: language %v: %w", language, speech.ErrUnsupported)
	}
	constrained := language == speech.Unknown && opts.Languages.Len() > 1
	t.limits = nil
	switch set := opts.Languages; {
	case language != speech.Unknown:
		set = speech.Languages(language)
		fallthrough
	case constrained:
		var err error
		if t.limits, err = t.m.limitsOf(set); err != nil {
			return err
		}
	}
	dst.Reset()
	for start := 0; start < len(pcm); {
		end := len(pcm)
		if end-start > maxChunkSamples {
			end = quietCut(pcm, start)
		}
		piece := pcm[start:end]
		if start > 0 || end < len(pcm) {
			if n := len(piece); n < minChunkSamples {
				t.pcm = grow(t.pcm, minChunkSamples)[:minChunkSamples]
				copy(t.pcm, piece)
				clear(t.pcm[n:])
				piece = t.pcm
			}
		}
		partial := opts.Partial
		if start > 0 || end < len(pcm) {
			partial = nil
		}
		lang, err := t.transcribe(ctx, piece, opts.Context, language, partial, constrained)
		if err != nil {
			return err
		}
		if opts.Segments && len(t.text) > 0 {
			dst.Segments = append(dst.Segments, speech.Segment{
				Start: float64(start) / sampleRate, End: float64(end) / sampleRate,
				TextStart: len(dst.Text), TextEnd: len(dst.Text) + len(t.text)})
		}
		dst.Text = append(dst.Text, t.text...)
		if dst.Language == speech.Unknown {
			dst.Language = lang
		}
		start = end
	}
	if opts.Turn && len(dst.Text) > 0 {
		// The state that ended the last piece's transcript has heard the
		// audio and read the words.
		dst.Turn = t.m.turn.predict(t.hidden)
	}
	return nil
}

// quietCut returns where to end the piece that starts at start: the
// quietest sample of the quietest 100 ms window within 5 s of the length
// limit, as the reference's split_audio_into_chunks chooses it.
func quietCut(pcm []float32, start int) int {
	cut := start + maxChunkSamples
	const expand, win = 5 * sampleRate, sampleRate / 10
	left, right := max(start, cut-expand), min(len(pcm), cut+expand)
	if right-left <= win {
		return cut
	}
	abs := func(v float32) float32 { return max(v, -v) }
	var sum float32
	for _, v := range pcm[left : left+win] {
		sum += abs(v)
	}
	best, bestSum := 0, sum
	for i := 1; left+i+win <= right; i++ {
		sum += abs(pcm[left+i+win-1]) - abs(pcm[left+i-1])
		if sum < bestSum {
			best, bestSum = i, sum
		}
	}
	inner, low := 0, abs(pcm[left+best])
	for i := 1; i < win; i++ {
		if v := abs(pcm[left+best+i]); v < low {
			inner, low = i, v
		}
	}
	return min(max(left+best+inner, start+1), len(pcm))
}

// transcribe runs one pass and leaves the text in t.text, returning the
// language. A partial transcript of the audio's beginning is checked and
// continued rather than decoded again.
func (t *Transcriber) transcribe(ctx context.Context, pcm []float32, context string, language speech.Language, partial *speech.Transcript, constrained bool) (speech.Language, error) {
	t.limit, t.forcedText = constrained, language != speech.Unknown
	t.text = t.text[:0]
	if err := ctx.Err(); err != nil {
		return speech.Unknown, err
	}
	// Decoded audio outside [-1, 1] is scaled down by its peak, as the
	// reference normalizes int-like input.
	var peak float32
	for _, v := range pcm {
		peak = max(peak, v, -v)
	}
	if peak > 1 {
		if len(t.pcm) == 0 || &t.pcm[0] != &pcm[0] {
			t.pcm = grow(t.pcm, len(pcm))[:len(pcm)]
			copy(t.pcm, pcm)
		}
		for i, v := range t.pcm {
			t.pcm[i] = min(max(v/peak, -1), 1)
		}
		pcm = t.pcm
	}
	frames, err := t.frontend.Frames(len(pcm))
	if err != nil || frames == 0 {
		return speech.Unknown, nil // too short to hold speech
	}
	e := t.m.enc
	n := e.freq[0] * frames
	if t.genc != nil {
		if t.features, err = t.genc.features(n); err != nil {
			return speech.Unknown, err
		}
	} else {
		t.features = grow(t.features, n)[:n]
	}
	if err := t.frontend.Into(pcm, t.features); err != nil {
		return speech.Unknown, fmt.Errorf("qwen3asr: features: %w", err)
	}
	if err := t.encodeContinuation(frames, partial != nil); err != nil {
		return speech.Unknown, err
	}
	if err := t.buildPrompt(context, language, e.tokens(frames)); err != nil {
		return speech.Unknown, err
	}
	if err := ctx.Err(); err != nil {
		return speech.Unknown, err
	}
	if err := t.draftFrom(partial, language); err != nil {
		return speech.Unknown, err
	}
	maxNew := max(minNewTokens, len(pcm)/sampleRate*newTokensPerSecond)
	if err := t.generate(ctx, maxNew); err != nil {
		return speech.Unknown, err
	}
	t.raw = t.m.tok.DecodeAppend(t.raw[:0], t.gen, true)
	return t.parse(language), nil
}

// encode runs the audio encoder on the features in t.features, leaving the
// embeddings in t.embeds.
func (t *Transcriber) encode(frames int) error {
	return t.encodeContinuation(frames, false)
}

func (t *Transcriber) encodeContinuation(frames int, continuing bool) error {
	defer runtime.KeepAlive(t)
	t.reusedAudioRows = 0
	t.audioRevision++
	if t.genc != nil {
		embeds, err := t.genc.encode(frames)
		if err != nil {
			return fmt.Errorf("qwen3asr: encoder: %w", err)
		}
		t.embeds = embeds
		return nil
	}
	e := t.m.enc
	n := e.tokens(frames) * e.out
	skip := 0
	if continuing {
		skip = t.encPrefix.reusable(e, t.features, frames)
	} else {
		t.encPrefix.reset()
	}
	kept := e.tokens(skip) * e.out
	if kept > len(t.embeds) {
		skip, kept = 0, 0
	}
	if cap(t.embeds) < n {
		dst := make([]float32, n)
		copy(dst, t.embeds[:kept])
		t.embeds = dst
	} else {
		t.embeds = t.embeds[:n]
	}
	_, err := e.encodeSuffix(t.features, frames, skip, t.embeds[kept:], t.enc)
	if err != nil {
		t.encPrefix.reset()
	} else if continuing {
		t.reusedAudioRows = kept / e.out
		t.encPrefix.remember(e, t.features, frames, skip > 0)
	}
	return err
}

// buildPrompt writes the chat prompt's token ids to t.ids with audio
// placeholder tokens, one per audio embedding.
func (t *Transcriber) buildPrompt(context string, language speech.Language, audio int) error {
	b := append(t.prompt[:0], "<|im_start|>system\n"...)
	b = append(b, context...)
	b = append(b, "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"...)
	if language != speech.Unknown {
		b = append(b, "language "...)
		b = append(b, language.Name()...)
		b = append(b, "<asr_text>"...)
	}
	t.prompt = b
	// Tokens never outnumber bytes; the audio placeholder expands below,
	// and a draft may follow.
	need := len(b) + audio + maxDraft
	if cap(t.ids) < need {
		t.ids = make([]int, 0, need)
	}
	// EncodeInto keeps no reference to the text after it returns.
	ids, err := t.m.tok.EncodeInto(unsafe.String(unsafe.SliceData(b), len(b)), t.ids[:0], &t.tokWS)
	if err != nil {
		return fmt.Errorf("qwen3asr: prompt: %w", err)
	}
	at := -1
	for i, id := range ids {
		if id == t.m.ids.audioPad {
			if at >= 0 {
				return errors.New("qwen3asr: the context holds an audio placeholder token")
			}
			at = i
		}
	}
	n := len(ids)
	ids = ids[:n+audio-1]
	copy(ids[at+audio:], ids[at+1:n])
	for i := range audio {
		ids[at+i] = t.m.ids.audioPad
	}
	t.ids = ids
	return nil
}

// draftFrom sets t.draft to the tokens the model wrote for partial: its
// text, after the language header unless the prompt forces the language.
// Without a partial transcript, or one with no language to continue, the
// draft is empty.
func (t *Transcriber) draftFrom(partial *speech.Transcript, forced speech.Language) error {
	t.draft = t.draft[:0]
	if partial == nil || len(partial.Text) == 0 || forced == speech.Unknown && !t.m.languages.Has(partial.Language) {
		return nil
	}
	b := t.prompt[:0]
	if forced == speech.Unknown {
		b = append(b, "language "...)
		b = append(b, partial.Language.Name()...)
		b = append(b, "<asr_text>"...)
	}
	b = append(b, partial.Text...)
	t.prompt = b
	if cap(t.draft) < len(b) {
		t.draft = make([]int, 0, len(b))
	}
	draft, err := t.m.tok.EncodeInto(unsafe.String(unsafe.SliceData(b), len(b)), t.draft, &t.tokWS)
	if err != nil {
		return fmt.Errorf("qwen3asr: partial transcript: %w", err)
	}
	t.draft = draft[:min(len(draft), maxDraft)]
	return nil
}

// generate decodes greedily from the prompt in t.ids into t.gen, stopping at
// an end token or after maxNew tokens. A draft in t.draft is checked in the
// prompt's pass: the tokens the model would write itself are kept, and
// decoding goes on from the first it would not. The key-value cache keeps
// the prompt prefix that does not depend on the audio for the next call.
func (t *Transcriber) generate(ctx context.Context, maxNew int) error {
	ev := t.m.eval
	prompt := len(t.ids)
	t.ids = append(t.ids, t.draft...)
	need := len(t.ids) + maxNew
	if t.kv == nil || t.kv.Capacity() < need {
		capacity := 1024
		for capacity < need {
			capacity *= 2
		}
		kv, err := ev.NewPrefixKV(min(capacity, t.m.lm.Config().MaxPositions))
		if err != nil {
			return fmt.Errorf("qwen3asr: %w", err)
		}
		_ = t.kv.Close()
		t.kv = kv
		t.decoderPrefix.reset()
	}
	keep := min(t.kv.CommonPrefix(t.ids), prompt-1)
	reuse := t.decoderPrefix.reusable(keep, t.reusedAudioRows, t.audioRevision)
	if t.ids[keep] != t.m.ids.audioPad {
		reuse = 0
	}
	t.decoderPrefix.reset() // a failed or partial prefill cannot advertise cached rows
	embeds := qwen3lm.Embeds{Token: t.m.ids.audioPad, Rows: t.embeds}
	t.gen = grow(t.gen, maxNew+maxDraft)
	done := false
	if len(t.draft) == 0 {
		if err := ev.HiddenLastContinueEmbedInto(t.kv, keep, reuse, t.ids[keep:], embeds, t.hidden, t.lm); err != nil {
			return fmt.Errorf("qwen3asr: prefill: %w", err)
		}
	} else {
		var err error
		done, err = t.verify(prompt, keep, reuse, embeds, maxNew)
		if err != nil {
			return err
		}
	}
	// The warm prompt anchor is the first audio placeholder. Cold prefills
	// use a different arithmetic anchor and are deliberately not reused.
	if t.ids[keep] == t.m.ids.audioPad {
		t.decoderPrefix.remember(keep, len(t.embeds)/len(t.hidden), t.audioRevision)
	}
	if done {
		return nil
	}
	for {
		if err := ev.LogitsInto(t.hidden, t.logits, t.lm); err != nil {
			return err
		}
		next := t.pick(t.logits, t.gen)
		if next == t.m.ids.eos[0] || next == t.m.ids.eos[1] {
			return nil
		}
		t.gen = append(t.gen, next)
		if len(t.gen) >= maxNew {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := ev.HiddenLastExtendInto(t.kv, len(t.kv.Tokens()), t.gen[len(t.gen)-1:], t.hidden, t.lm); err != nil {
			return fmt.Errorf("qwen3asr: decode: %w", err)
		}
	}
}

// verify evaluates the prompt and the draft after it in one pass, keeps the
// draft's longest prefix the model agrees with, and appends the model's own
// token after it. It reports whether that ends the transcript; otherwise
// t.hidden is the state after the appended token.
func (t *Transcriber) verify(prompt, keep, reuse int, embeds qwen3lm.Embeds, maxNew int) (bool, error) {
	ev, h, vocab := t.m.eval, len(t.hidden), len(t.logits)
	k := len(t.draft) + 1 // the prompt's last position, then each draft token's
	t.tail = grow(t.tail, k*h)[:k*h]
	if err := ev.HiddenTailContinueEmbedInto(t.kv, keep, reuse, t.ids[keep:], embeds, t.tail, t.lm); err != nil {
		return false, fmt.Errorf("qwen3asr: prefill: %w", err)
	}
	t.vlogits = grow(t.vlogits, verifyRows*vocab)
	agreed, next := -1, 0
	for r0 := 0; r0 < k && agreed < 0; r0 += verifyRows {
		n := min(verifyRows, k-r0)
		logits := t.vlogits[:n*vocab]
		if err := ev.LogitsRowsInto(t.tail[r0*h:(r0+n)*h], logits, t.lm); err != nil {
			return false, err
		}
		for j := range n {
			if a := t.pick(logits[j*vocab:(j+1)*vocab], t.draft[:r0+j]); r0+j == len(t.draft) || a != t.draft[r0+j] {
				agreed, next = r0+j, a
				break
			}
		}
	}
	t.gen = append(t.gen, t.draft[:agreed]...)
	// The state that chose next, as generate leaves it: when next ends the
	// transcript, it judges the turn.
	copy(t.hidden, t.tail[agreed*h:(agreed+1)*h])
	if next == t.m.ids.eos[0] || next == t.m.ids.eos[1] {
		return true, nil
	}
	t.gen = append(t.gen, next)
	if len(t.gen) >= maxNew {
		return true, nil
	}
	// The cache holds the whole draft: continue after the agreed part.
	if err := ev.HiddenLastExtendInto(t.kv, prompt+agreed, t.gen[len(t.gen)-1:], t.hidden, t.lm); err != nil {
		return false, fmt.Errorf("qwen3asr: decode: %w", err)
	}
	return false, nil
}

// pick chooses the token after before: the likeliest; where the output
// names its language and detection is limited, the likeliest allowed name;
// and in the transcript of given languages, the likeliest in their scripts.
func (t *Transcriber) pick(logits []float32, before []int) int {
	switch {
	case t.limit && len(before) == 1 && before[0] == t.m.ids.language:
		best := t.limits.names[0]
		for _, id := range t.limits.names[1:] {
			if logits[id] > logits[best] {
				best = id
			}
		}
		return best
	case t.limits != nil && t.limits.mask != nil && (t.forcedText || slices.Contains(before, t.m.ids.asrText)):
		mask := t.limits.mask
		best := -1
		for id, v := range logits {
			if mask[id] && (best < 0 || v > logits[best]) {
				best = id
			}
		}
		return best
	}
	return argmax(logits)
}

// argmax returns the first index of the largest value.
func argmax(values []float32) int {
	best, at := values[0], 0
	for i, v := range values {
		if v > best {
			best, at = v, i
		}
	}
	return at
}

// grow returns s emptied, with capacity for at least n elements.
func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, 0, n)
	}
	return s[:0]
}
