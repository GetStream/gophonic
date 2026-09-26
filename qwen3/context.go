// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/GetStream/gophonic/internal/qwen3lm"
)

// Context holds a long shared text, such as a conversation, that several
// multiple-choice questions are asked about. Set evaluates the text once and
// keeps its keys and values, re-evaluating only tokens that changed since the
// previous Set, and Ask answers any number of ContextQuestions as short
// sequences against it in shared forward passes.
//
// The prompt puts the context first: "<context>\n\n<question>\n<options>".
// On our probes this agrees with Question for classification tasks
// (sentiment, intent, tools, genre) but not for turn detection, which needs
// Question's input-last layout. Validate each question on your own data.
// A Context is not safe for concurrent use.
type Context struct {
	m      *encoder
	kv     *qwen3lm.PrefixKV
	header []int  // "<|im_start|>user\n"
	text   []byte // context text plus separator, tokenized together
	ids    []int  // header, context, and separator tokens
	hidden [][]float32
	seqs   [][]int
	tok    TokenizerWorkspace
}

// ContextQuestion is a question prepared for asking about a Context. It is
// independent of any particular Context and may be reused across them.
type ContextQuestion struct {
	tail    []int // question, lettered options, footer, and answer prefix
	options int
}

const (
	contextSeparator = "\n\n"
	contextFooter    = "Answer with the letter only.<|im_end|>\n"
)

// NewContext allocates a context of up to maxTokens tokens (288 KiB of keys
// and values per token).
func (e *encoder) NewContext(maxTokens int) (*Context, error) {
	if e == nil || e.letters == nil {
		return nil, errors.New("qwen3: this model was opened without the language-model head")
	}
	if maxTokens < 1 || maxTokens > e.model.Config().MaxPositions {
		return nil, fmt.Errorf("qwen3: context size %d outside [1,%d]", maxTokens, e.model.Config().MaxPositions)
	}
	kv, err := e.eval.NewPrefixKV(maxTokens)
	if err != nil {
		return nil, err
	}
	c := &Context{m: e, kv: kv, ids: make([]int, 0, 4*maxTokens), hidden: [][]float32{make([]float32, e.model.Config().Hidden)}}
	// The header ends with a special token and a newline, a pre-tokenizer
	// boundary. The separator is tokenized together with the text, because
	// the pre-tokenizer joins trailing newlines to preceding punctuation.
	if c.header, err = e.tokens.EncodeInto("<|im_start|>user\n", make([]int, 0, 8), &c.tok); err != nil {
		return nil, err
	}
	c.text = make([]byte, 0, 4*maxTokens)
	return c, nil
}

// ContextQuestion prepares a question with 2 to 26 options for Context.Ask.
func (e *encoder) ContextQuestion(question string, options []string) (*ContextQuestion, error) {
	if e == nil || e.letters == nil {
		return nil, errors.New("qwen3: this model was opened without the language-model head")
	}
	if len(options) < 2 || len(options) > maxChoices {
		return nil, fmt.Errorf("qwen3: %d options, want 2 to %d", len(options), maxChoices)
	}
	text := strings.TrimSpace(question) + "\n"
	for i, o := range options {
		text += string(rune('A'+i)) + ") " + o + "\n"
	}
	text += contextFooter + e.answer
	var ws TokenizerWorkspace
	tail, err := e.tokens.EncodeInto(text, make([]int, 0, len(text)), &ws)
	if err != nil {
		return nil, err
	}
	return &ContextQuestion{tail: tail, options: len(options)}, nil
}

// Set replaces the context text. Only tokens after the longest common prefix
// with the previous text are evaluated, so appending to a conversation costs
// the new turn alone. Warmed calls allocate nothing.
func (c *Context) Set(ctx context.Context, text string) error {
	if c == nil {
		return errors.New("qwen3: nil context")
	}
	text = strings.TrimSpace(text)
	if len(text)+len(contextSeparator) > cap(c.text) || len(c.header)+len(text)+len(contextSeparator) > cap(c.ids) {
		return fmt.Errorf("qwen3: context text of %d bytes exceeds its buffer", len(text))
	}
	c.text = append(append(c.text[:0], text...), contextSeparator...)
	ids := append(c.ids[:0], c.header...)
	// The string aliases c.text only while EncodeInto runs.
	body, err := c.m.tokens.EncodeInto(unsafe.String(unsafe.SliceData(c.text), len(c.text)), ids[len(ids):len(ids):cap(ids)], &c.tok)
	if err != nil {
		return err
	}
	ids = ids[:len(ids)+len(body)]
	c.ids = ids
	if len(ids) > c.kv.Capacity() {
		return fmt.Errorf("qwen3: context of %d tokens exceeds its limit %d", len(ids), c.kv.Capacity())
	}
	keep := min(c.kv.CommonPrefix(ids), len(ids)-1)
	if keep == len(ids)-1 && len(c.kv.Tokens()) == len(ids) {
		return nil // unchanged
	}
	e := c.m
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("qwen3: model closed")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return e.eval.HiddenLastExtendInto(c.kv, keep, ids[keep:], c.hidden[0], e.ws)
}

// Ask answers every question about the current context, writing question i's
// option probabilities to probs[i]. Questions share forward passes. Buffers
// grow to the largest set asked; later calls up to that size allocate
// nothing.
func (c *Context) Ask(ctx context.Context, questions []*ContextQuestion, probs [][]float32) error {
	if c == nil {
		return errors.New("qwen3: nil context")
	}
	if len(c.kv.Tokens()) == 0 {
		return errors.New("qwen3: context is empty; call Set first")
	}
	if len(probs) != len(questions) {
		return fmt.Errorf("qwen3: %d probability rows for %d questions", len(probs), len(questions))
	}
	for i, q := range questions {
		if len(probs[i]) != q.options {
			return fmt.Errorf("qwen3: row %d has %d probabilities for %d options", i, len(probs[i]), q.options)
		}
		if len(c.kv.Tokens())+len(q.tail) > maxTokens {
			return fmt.Errorf("qwen3: question %d does not fit after the context", i)
		}
	}
	e := c.m
	if len(c.seqs) < len(questions) {
		c.seqs = make([][]int, len(questions))
		hidden := make([]float32, len(questions)*e.model.Config().Hidden)
		c.hidden = make([][]float32, len(questions))
		for i := range c.hidden {
			c.hidden[i] = hidden[i*e.model.Config().Hidden : (i+1)*e.model.Config().Hidden]
		}
	}
	for i, q := range questions {
		c.seqs[i] = q.tail
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("qwen3: model closed")
	}
	seqs := c.seqs[:len(questions)]
	for start := 0; start < len(seqs); {
		if err := ctx.Err(); err != nil {
			return err
		}
		end, tokens := start, 0
		for end < len(seqs) && (end == start || tokens+len(seqs[end]) <= batchTokens) {
			tokens += len(seqs[end])
			end++
		}
		if err := e.eval.HiddenLastSharedInto(c.kv, seqs[start:end], c.hidden[start:end], e.ws); err != nil {
			return err
		}
		start = end
	}
	for i := range seqs {
		e.letterProbs(c.hidden[i], probs[i])
	}
	return nil
}
