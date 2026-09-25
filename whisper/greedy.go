// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"math"
	"strings"

	"github.com/GetStream/gophonic/internal/nn"
)

var (
	ErrGreedyNilPolicy      = errors.New("whisper: nil greedy policy")
	ErrGreedyLogitsShape    = errors.New("whisper: greedy logits are smaller than the vocabulary")
	ErrGreedyOutputTooSmall = errors.New("whisper: greedy logits output is smaller than the vocabulary")
	ErrGreedyPromptTooLong  = errors.New("whisper: greedy prompt exceeds the text context")
	ErrGreedyTokenRange     = errors.New("whisper: greedy prompt token is outside the vocabulary")
	ErrGreedyNoCandidate    = errors.New("whisper: no token remains after greedy filtering")
)

// GreedyOptions configures Whisper's deterministic, temperature-zero decoder.
// The zero value follows the reference filters: suppress blank and non-speech
// tokens, use the transcribe task, and limit the first timestamp to one second.
// Set WithoutTimestamps to add NoTimestamps to the initial context and skip
// timestamp rules.
type GreedyOptions struct {
	Language string
	Task     string

	WithoutTimestamps bool
	SuppressTokens    []int

	DisableBlankSuppression      bool
	DisableNonSpeechSuppression  bool
	DisableTimestampRules        bool
	DisableInitialTimestampLimit bool
	MaxInitialTimestampSeconds   float64
}

// GreedyPolicy applies the pinned OpenAI Whisper suppression and timestamp
// rules before selecting the first maximum logit. It owns immutable suppression
// tables and the sample-begin position recorded by PromptInto. Give each worker
// its own policy when prompts differ; Tokenizer values themselves are safe to
// share between workers.
type GreedyPolicy struct {
	tokenizer                *Tokenizer
	options                  GreedyOptions
	suppressed               []bool
	suppressedIDs            []int // the true entries of suppressed, in order
	sampleBegin              int
	spaceToken               int
	withoutTimestamps        bool
	maxInitialTimestampIndex int
	initialTimestampLimit    bool
}

// NewGreedyPolicy builds the token suppression table once. Its select path
// performs no heap allocations.
func NewGreedyPolicy(tokenizer *Tokenizer, options GreedyOptions) (*GreedyPolicy, error) {
	if tokenizer == nil {
		return nil, ErrTokenizerNil
	}
	p := &GreedyPolicy{
		tokenizer:         tokenizer,
		options:           options,
		suppressed:        make([]bool, tokenizer.VocabSize()),
		withoutTimestamps: options.WithoutTimestamps,
	}
	if tokenizer.Kind() == Multilingual {
		language := strings.ToLower(options.Language)
		if language == "" {
			language = "en"
		}
		if _, ok := tokenizer.LanguageToken(language); !ok {
			return nil, errors.New("whisper: unsupported greedy language")
		}
		if options.Task != "" && options.Task != "transcribe" && options.Task != "translate" {
			return nil, errors.New("whisper: unsupported greedy task")
		}
	} else if options.Task != "" && options.Task != "transcribe" && options.Task != "translate" {
		return nil, errors.New("whisper: unsupported greedy task")
	}

	// The pinned reference always suppresses control tokens that are not valid
	// continuations. NoTimestamps itself remains selectable; timestamp tokens
	// also remain unmasked when without_timestamps skips timestamp rules.
	for _, id := range []int{tokenizer.Transcribe(), tokenizer.Translate(), tokenizer.SOT(), tokenizer.SOTPrev(), tokenizer.SOTLM(), tokenizer.NoSpeech()} {
		p.suppressed[id] = true
	}
	if !options.DisableNonSpeechSuppression {
		ids, err := nonSpeechTokenIDs(tokenizer)
		if err != nil {
			return nil, err
		}
		for _, id := range ids {
			p.suppressed[id] = true
		}
	}
	for _, id := range options.SuppressTokens {
		if id < 0 || id >= tokenizer.VocabSize() {
			return nil, ErrGreedyTokenRange
		}
		p.suppressed[id] = true
	}

	space := make([]int, 0, 8)
	space, err := tokenizer.EncodeInto(space, " ")
	if err != nil {
		return nil, err
	}
	if len(space) != 1 {
		return nil, errors.New("whisper: space must encode as one token")
	}
	p.spaceToken = space[0]
	p.maxInitialTimestampIndex = 50 // one second / 20 ms timestamp precision
	if options.MaxInitialTimestampSeconds > 0 {
		p.maxInitialTimestampIndex = int(math.Round(options.MaxInitialTimestampSeconds / 0.02))
	}
	p.initialTimestampLimit = !options.DisableInitialTimestampLimit && !p.withoutTimestamps
	for id, suppress := range p.suppressed {
		if suppress {
			p.suppressedIDs = append(p.suppressedIDs, id)
		}
	}
	return p, nil
}

// PromptInto appends the initial decoder context to dst and returns its final
// length. prompt contains previous-window context tokens; prefix contains fixed
// text tokens for the current window. Both are already tokenized. EnglishOnly
// starts with [SOT] unless WithoutTimestamps is set.
func (p *GreedyPolicy) PromptInto(dst []int, prompt, prefix []int) (int, error) {
	if p == nil {
		return len(dst), ErrGreedyNilPolicy
	}
	if len(prompt) > TextContext/2-1 {
		prompt = prompt[len(prompt)-(TextContext/2-1):]
	}
	need := 1 // SOT
	if len(prompt) != 0 {
		need += 1 + len(prompt) // SOTPrev and previous text
	}
	if p.tokenizer.Kind() == Multilingual {
		need += 2 // language and task
	}
	if p.withoutTimestamps {
		need++
	}
	need += len(prefix)
	if need > TextContext || len(dst)+need > cap(dst) {
		if need > TextContext {
			return len(dst), ErrGreedyPromptTooLong
		}
		return len(dst), ErrGreedyOutputTooSmall
	}
	for _, id := range prompt {
		if id < 0 || id >= p.tokenizer.VocabSize() {
			return len(dst), ErrGreedyTokenRange
		}
	}
	for _, id := range prefix {
		if id < 0 || id >= p.tokenizer.VocabSize() {
			return len(dst), ErrGreedyTokenRange
		}
	}
	if len(prompt) != 0 {
		dst = append(dst, p.tokenizer.SOTPrev())
		dst = append(dst, prompt...)
	}
	dst = append(dst, p.tokenizer.SOT())
	if p.tokenizer.Kind() == Multilingual {
		language := strings.ToLower(p.options.Language)
		if language == "" {
			language = "en"
		}
		languageID, _ := p.tokenizer.LanguageToken(language)
		dst = append(dst, languageID)
		task := p.options.Task
		if task == "" {
			task = "transcribe"
		}
		if task == "translate" {
			dst = append(dst, p.tokenizer.Translate())
		} else {
			dst = append(dst, p.tokenizer.Transcribe())
		}
	}
	if p.withoutTimestamps {
		dst = append(dst, p.tokenizer.NoTimestamps())
	}
	dst = append(dst, prefix...)
	p.sampleBegin = len(dst)
	return p.sampleBegin, nil
}

// SelectNextInto copies logits into dst, applies Whisper's filters in place,
// then returns the first token with the maximum remaining logit. dst may alias
// logits. history is the current prompt plus generated IDs. It must include the
// complete context written by the most recent PromptInto call.
func (p *GreedyPolicy) SelectNextInto(dst, logits []float32, history []int) (int, error) {
	if p == nil {
		return 0, ErrGreedyNilPolicy
	}
	vocab := p.tokenizer.VocabSize()
	if len(logits) < vocab {
		return 0, ErrGreedyLogitsShape
	}
	if len(dst) < vocab {
		return 0, ErrGreedyOutputTooSmall
	}
	if &dst[0] != &logits[0] {
		copy(dst[:vocab], logits[:vocab])
	}
	for _, id := range p.suppressedIDs {
		dst[id] = float32(math.Inf(-1))
	}
	if !p.options.DisableBlankSuppression && len(history) == p.sampleBegin {
		dst[p.tokenizer.EOT()] = float32(math.Inf(-1))
		dst[p.spaceToken] = float32(math.Inf(-1))
	}
	if !p.withoutTimestamps && !p.options.DisableTimestampRules {
		p.applyTimestampRules(dst[:vocab], history)
	}
	bestID := argmaxFinite(dst[:vocab])
	if bestID < 0 {
		return 0, ErrGreedyNoCandidate
	}
	return bestID, nil
}

func (p *GreedyPolicy) applyTimestampRules(logits []float32, history []int) {
	t := p.tokenizer
	begin := t.TimestampBegin()
	if t.NoTimestamps() < len(logits) {
		logits[t.NoTimestamps()] = float32(math.Inf(-1))
	}
	seq := history
	if p.sampleBegin <= len(history) {
		seq = history[p.sampleBegin:]
	}
	lastTimestamp := len(seq) > 0 && seq[len(seq)-1] >= begin
	penultimateTimestamp := len(seq) < 2 || seq[len(seq)-2] >= begin
	if lastTimestamp {
		if penultimateTimestamp {
			maskLogits(logits, begin, len(logits))
		} else {
			maskLogits(logits, 0, t.EOT())
		}
	}
	lastSeen := -1
	for _, id := range seq {
		if id >= begin {
			lastSeen = id
		}
	}
	if lastSeen >= 0 {
		lastAllowed := lastSeen + 1
		if lastTimestamp && !penultimateTimestamp {
			lastAllowed = lastSeen
		}
		maskLogits(logits, begin, lastAllowed)
	}
	if len(history) == p.sampleBegin {
		maskLogits(logits, 0, begin)
		if p.initialTimestampLimit {
			lastAllowed := begin + p.maxInitialTimestampIndex
			if lastAllowed+1 < len(logits) {
				maskLogits(logits, lastAllowed+1, len(logits))
			}
		}
	}
	maxText := float64(math.Inf(-1))
	maxTime := float64(math.Inf(-1))
	for _, value := range logits[:begin] {
		if !math.IsNaN(float64(value)) && float64(value) > maxText {
			maxText = float64(value)
		}
	}
	for _, value := range logits[begin:] {
		if !math.IsNaN(float64(value)) && float64(value) > maxTime {
			maxTime = float64(value)
		}
	}
	if !math.IsInf(maxTime, -1) {
		var sum float64
		for _, value := range logits[begin:] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), -1) {
				continue
			}
			sum += math.Exp(float64(value) - maxTime)
		}
		if maxTime+math.Log(sum) > maxText {
			maskLogits(logits, 0, begin)
		}
	}
}

func maskLogits(logits []float32, start, end int) {
	if start < 0 {
		start = 0
	}
	if end > len(logits) {
		end = len(logits)
	}
	for i := start; i < end; i++ {
		logits[i] = float32(math.Inf(-1))
	}
}

func nonSpeechTokenIDs(tokenizer *Tokenizer) ([]int, error) {
	ids := make([]int, 0, 96)
	seen := make([]bool, tokenizer.VocabSize())
	add := func(id int) {
		if id >= 0 && id < len(seen) && !seen[id] {
			seen[id] = true
			ids = append(ids, id)
		}
	}
	encodeFirst := func(text string) ([]int, error) {
		encoded := make([]int, 0, 32)
		return tokenizer.EncodeInto(encoded, text)
	}
	for _, text := range []string{" -", " '"} {
		encoded, err := encodeFirst(text)
		if err != nil {
			return nil, err
		}
		if len(encoded) != 0 {
			add(encoded[0])
		}
	}
	symbols := []rune("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")
	multi := strings.Fields("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪")
	for _, symbol := range symbols {
		if _, err := suppressSymbol(tokenizer, string(symbol), false, add); err != nil {
			return nil, err
		}
	}
	for _, text := range multi {
		if _, err := suppressSymbol(tokenizer, text, false, add); err != nil {
			return nil, err
		}
	}
	for _, symbol := range []rune("♩♪♫♬♭♮♯") {
		if _, err := suppressSymbol(tokenizer, string(symbol), true, add); err != nil {
			return nil, err
		}
	}
	for _, symbol := range symbols {
		if _, err := suppressSymbol(tokenizer, " "+string(symbol), false, add); err != nil {
			return nil, err
		}
	}
	for _, text := range multi {
		if _, err := suppressSymbol(tokenizer, " "+text, false, add); err != nil {
			return nil, err
		}
	}
	for _, symbol := range []rune("♩♪♫♬♭♮♯") {
		if _, err := suppressSymbol(tokenizer, " "+string(symbol), true, add); err != nil {
			return nil, err
		}
	}
	// The reference returns these IDs sorted and unique.
	for i := 1; i < len(ids); i++ {
		for j := i; j > 0 && ids[j] < ids[j-1]; j-- {
			ids[j], ids[j-1] = ids[j-1], ids[j]
		}
	}
	return ids, nil
}

func suppressSymbol(tokenizer *Tokenizer, text string, always bool, add func(int)) (bool, error) {
	ids := make([]int, 0, 32)
	encoded, err := tokenizer.EncodeInto(ids, text)
	if err != nil {
		return false, err
	}
	if len(encoded) == 0 {
		return false, nil
	}
	if len(encoded) == 1 || always {
		add(encoded[0])
	}
	return true, nil
}

// argmaxFinite returns the first index of the largest value that is neither
// NaN nor -Inf, or -1 when there is none.
func argmaxFinite(values []float32) int {
	n := len(values) / 16 * 16
	if nn.Accelerated && n > 0 {
		best := maxNumNEON(&values[0], n)
		for _, v := range values[n:] {
			if v > best || best != best {
				best = v
			}
		}
		if best != best || math.IsInf(float64(best), -1) {
			return -1
		}
		for id, v := range values {
			if v == best {
				return id
			}
		}
	}
	// NaN and -Inf never compare greater than -Inf, so they are skipped;
	// strict comparison keeps the first of equal maxima.
	bestID := -1
	best := float32(math.Inf(-1))
	for id, value := range values {
		if value > best {
			bestID, best = id, value
		}
	}
	return bestID
}
