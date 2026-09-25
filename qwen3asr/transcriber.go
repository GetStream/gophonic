// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"errors"
	"fmt"
	"runtime"
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
// Calls must not overlap; open one Transcriber per concurrent lane. Warm
// calls on audio no longer than an earlier call's allocate nothing.
type Transcriber struct {
	m        *Model
	frontend *mel.Spectrogram
	enc      *encoderWorkspace    // CPU encoder
	genc     *gpuEncoderWorkspace // GPU encoder, when the model has one
	lm       *qwen3lm.Workspace
	kv       *qwen3lm.PrefixKV
	tokWS    qwen3lm.TokenizerWorkspace

	pcm      []float32 // normalized or padded copy of the input, when needed
	features []float32
	embeds   []float32
	prompt   []byte
	ids      []int
	hidden   []float32
	logits   []float32
	gen      []int
	draft    []int     // a partial transcript's tokens
	tail     []float32 // the states that check the draft
	vlogits  []float32 // their logits, verifyRows at a time
	raw      []byte
	runes    []rune
	fixed    []rune
	text     []byte
	closed   bool
}

// NewTranscriber opens a lane over m with workers CPU workers, including
// the caller; zero picks GOMAXPROCS, at most 8 for the encoder.
func NewTranscriber(m *Model, workers int) (*Transcriber, error) {
	if m == nil {
		return nil, errors.New("qwen3asr: nil model")
	}
	if workers < 0 || workers > 64 {
		return nil, fmt.Errorf("qwen3asr: invalid worker count %d", workers)
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
	return t.lm.Close()
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
	language := ""
	if opts.Language != "" {
		name, ok := speech.LanguageName(opts.Language)
		if !ok || !t.m.languages[name] {
			return fmt.Errorf("qwen3asr: language %q: %w", opts.Language, speech.ErrUnsupported)
		}
		language = name
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
		lang, err := t.transcribe(ctx, piece, opts.Context, language, partial)
		if err != nil {
			return err
		}
		if opts.Segments && len(t.text) > 0 {
			dst.Segments = append(dst.Segments, speech.Segment{
				Start: float64(start) / sampleRate, End: float64(end) / sampleRate,
				TextStart: len(dst.Text), TextEnd: len(dst.Text) + len(t.text)})
		}
		dst.Text = append(dst.Text, t.text...)
		if dst.Language == "" {
			dst.Language = lang
		}
		start = end
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
func (t *Transcriber) transcribe(ctx context.Context, pcm []float32, context, language string, partial *speech.Transcript) (string, error) {
	t.text = t.text[:0]
	if err := ctx.Err(); err != nil {
		return "", err
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
		return "", nil // too short to hold speech
	}
	e := t.m.enc
	n := e.freq[0] * frames
	if t.genc != nil {
		if t.features, err = t.genc.features(n); err != nil {
			return "", err
		}
	} else {
		t.features = grow(t.features, n)[:n]
	}
	if err := t.frontend.Into(pcm, t.features); err != nil {
		return "", fmt.Errorf("qwen3asr: features: %w", err)
	}
	if err := t.encode(frames); err != nil {
		return "", err
	}
	if err := t.buildPrompt(context, language, e.tokens(frames)); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := t.draftFrom(partial, language); err != nil {
		return "", err
	}
	maxNew := max(minNewTokens, len(pcm)/sampleRate*newTokensPerSecond)
	if err := t.generate(ctx, maxNew); err != nil {
		return "", err
	}
	t.raw = t.m.tok.DecodeAppend(t.raw[:0], t.gen, true)
	return t.parse(language), nil
}

// encode runs the audio encoder on the features in t.features, leaving the
// embeddings in t.embeds.
func (t *Transcriber) encode(frames int) error {
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
	t.embeds = grow(t.embeds, n)[:n]
	_, err := e.encode(t.features, frames, t.embeds, t.enc)
	return err
}

// buildPrompt writes the chat prompt's token ids to t.ids with audio
// placeholder tokens, one per audio embedding.
func (t *Transcriber) buildPrompt(context, language string, audio int) error {
	b := append(t.prompt[:0], "<|im_start|>system\n"...)
	b = append(b, context...)
	b = append(b, "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"...)
	if language != "" {
		b = append(b, "language "...)
		b = append(b, language...)
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
func (t *Transcriber) draftFrom(partial *speech.Transcript, forced string) error {
	t.draft = t.draft[:0]
	if partial == nil || len(partial.Text) == 0 || forced == "" && !t.m.languages[partial.Language] {
		return nil
	}
	b := t.prompt[:0]
	if forced == "" {
		b = append(b, "language "...)
		b = append(b, partial.Language...)
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
		t.kv = kv
	}
	keep := min(t.kv.CommonPrefix(t.ids), prompt-1)
	embeds := qwen3lm.Embeds{Token: t.m.ids.audioPad, Rows: t.embeds}
	t.gen = grow(t.gen, maxNew+maxDraft)
	if len(t.draft) == 0 {
		if err := ev.HiddenLastExtendEmbedInto(t.kv, keep, t.ids[keep:], embeds, t.hidden, t.lm); err != nil {
			return fmt.Errorf("qwen3asr: prefill: %w", err)
		}
	} else {
		done, err := t.verify(prompt, keep, embeds, maxNew)
		if err != nil || done {
			return err
		}
	}
	for {
		if err := ev.LogitsInto(t.hidden, t.logits, t.lm); err != nil {
			return err
		}
		next := argmax(t.logits)
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
func (t *Transcriber) verify(prompt, keep int, embeds qwen3lm.Embeds, maxNew int) (bool, error) {
	ev, h, vocab := t.m.eval, len(t.hidden), len(t.logits)
	k := len(t.draft) + 1 // the prompt's last position, then each draft token's
	t.tail = grow(t.tail, k*h)[:k*h]
	if err := ev.HiddenTailExtendEmbedInto(t.kv, keep, t.ids[keep:], embeds, t.tail, t.lm); err != nil {
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
			if a := argmax(logits[j*vocab : (j+1)*vocab]); r0+j == len(t.draft) || a != t.draft[r0+j] {
				agreed, next = r0+j, a
				break
			}
		}
	}
	t.gen = append(t.gen, t.draft[:agreed]...)
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
