// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/GetStream/gophonic/internal/q8gemm"
)

// maxChoices is the number of answer letters Choose can offer (A–Z).
const maxChoices = 26

// letterHead holds the language-model head rows of the answer letters A–Z,
// so Choose scores options without loading the full 152k-token head.
type letterHead struct {
	rows []float32 // [26][hidden]
}

func loadLetterHead(dir string, tokens *Tokenizer, hidden, vocab int) (*letterHead, error) {
	ids := make([]int, maxChoices)
	var ws TokenizerWorkspace
	for i := range ids {
		got, err := tokens.EncodeInto(string(rune('A'+i)), make([]int, 0, 4), &ws)
		if err != nil || len(got) != 1 {
			return nil, fmt.Errorf("qwen3: answer letter %c is not a single token", 'A'+i)
		}
		ids[i] = got[0]
	}
	st, err := openSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer st.close()
	t, err := st.lookup("lm_head.weight", vocab, hidden)
	if err != nil {
		return nil, err
	}
	if t.dtype != "BF16" {
		return nil, fmt.Errorf("qwen3: lm_head.weight is %s; the loader expects BF16", t.dtype)
	}
	h := &letterHead{rows: make([]float32, maxChoices*hidden)}
	raw := make([]uint16, hidden)
	for i, id := range ids {
		row := t
		row.offset += int64(id) * int64(hidden) * 2
		if err := readInto(row, raw); err != nil {
			return nil, err
		}
		for j, b := range raw {
			h.rows[i*hidden+j] = q8gemm.BF16ToF32(b)
		}
	}
	return h, nil
}

// Question is a prepared multiple-choice question. Model.Question tokenizes
// the fixed parts of the prompt once; each Choose then tokenizes only its
// input and assembles token IDs directly, so the hot path builds no strings
// and does no map lookups. A Question is not safe for concurrent use; prepare
// one per goroutine (calls on its Model are serialized anyway).
type Question struct {
	m       *Model
	prefix  []int // chat header, question, lettered options, "Input:\n"
	suffix  []int // end of turn and the empty non-thinking block
	options int
	input   []int // reusable input token buffer
	prompt  [1][]int
	hidden  [1][]float32
	tok     TokenizerWorkspace
}

// The prompt ends each fixed part at a pre-tokenizer boundary: the prefix
// with a newline and the suffix with a special token. Tokenizing the parts
// separately therefore yields exactly the IDs of the whole prompt, provided
// the input has no leading or trailing white space (Choose trims it).
const (
	questionHeader = "<|im_start|>user\n"
	questionFooter = "Answer with the letter only.\n\nInput:\n"
	questionSuffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

// Question prepares a multiple-choice question with 2 to 26 options. It
// allocates; reuse the result for every input.
func (e *Model) Question(question string, options []string) (*Question, error) {
	if e == nil || e.letters == nil {
		return nil, errors.New("qwen3: this model was opened without the language-model head")
	}
	if len(options) < 2 || len(options) > maxChoices {
		return nil, fmt.Errorf("qwen3: %d options, want 2 to %d", len(options), maxChoices)
	}
	text := questionHeader + question + "\n"
	for i, o := range options {
		text += string(rune('A'+i)) + ") " + o + "\n"
	}
	text += questionFooter
	q := &Question{m: e, options: len(options)}
	var err error
	if q.prefix, err = e.tokens.EncodeInto(text, make([]int, 0, len(text)), &q.tok); err != nil {
		return nil, err
	}
	if q.suffix, err = e.tokens.EncodeInto(questionSuffix, make([]int, 0, len(questionSuffix)), &q.tok); err != nil {
		return nil, err
	}
	if len(q.prefix)+len(q.suffix) >= maxTokens {
		return nil, fmt.Errorf("qwen3: question uses %d of the %d-token limit", len(q.prefix)+len(q.suffix), maxTokens)
	}
	q.hidden[0] = make([]float32, e.model.cfg.hidden)
	return q, nil
}

// Choose writes the model's probability for each option, in order, that it
// is the answer for input. Leading and trailing white space in input is
// ignored. Warmed calls allocate nothing.
func (q *Question) Choose(ctx context.Context, input string, probs []float32) error {
	if q == nil {
		return errors.New("qwen3: nil question")
	}
	input = strings.TrimSpace(input)
	if len(input) > cap(q.input) {
		q.input = make([]int, 0, len(input))
	}
	ids, err := q.m.tokens.EncodeInto(input, q.input[:0], &q.tok)
	if err != nil {
		return err
	}
	q.input = ids
	return q.ChooseTokens(ctx, ids, probs)
}

// ChooseTokens is Choose for input already tokenized with the model's
// tokenizer; it involves no strings at all. If the prompt would exceed the
// 2048-token limit, the input's earliest tokens are dropped.
func (q *Question) ChooseTokens(ctx context.Context, input []int, probs []float32) error {
	if q == nil {
		return errors.New("qwen3: nil question")
	}
	if len(probs) != q.options {
		return fmt.Errorf("qwen3: %d probabilities for %d options", len(probs), q.options)
	}
	if room := maxTokens - len(q.prefix) - len(q.suffix); len(input) > room {
		input = input[len(input)-room:]
	}
	p := append(append(append(q.prompt[0][:0], q.prefix...), input...), q.suffix...)
	q.prompt[0] = p
	if err := q.m.EmbedTokensInto(ctx, q.prompt[:], q.hidden[:]); err != nil {
		return err
	}
	hidden, rows := q.hidden[0], q.m.letters.rows
	maxLogit := math.Inf(-1)
	for i := range probs {
		probs[i] = dot32(hidden, rows[i*len(hidden):(i+1)*len(hidden)])
		maxLogit = max(maxLogit, float64(probs[i]))
	}
	var sum float64
	for i, l := range probs {
		p := math.Exp(float64(l) - maxLogit)
		probs[i] = float32(p)
		sum += p
	}
	for i := range probs {
		probs[i] = float32(float64(probs[i]) / sum)
	}
	return nil
}
