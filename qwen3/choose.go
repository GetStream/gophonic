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
// the fixed prompt once and evaluates its prefix (chat header, question, and
// lettered options) once, keeping that prefix's keys and values. Choose and
// ChooseBatch then evaluate only each input and the short prompt suffix
// against the stored prefix, several inputs per forward pass. The hot path
// builds no strings and does no map lookups. A Question is not safe for
// concurrent use; calls on its Model are serialized anyway.
type Question struct {
	m       *Model
	kv      *PrefixKV // the prompt prefix's keys and values, never modified
	suffix  []int     // end of turn and the empty non-thinking block
	options int
	ids     []int // token storage for a batch's inputs and suffixes
	seqs    [][]int
	hidden  [][]float32
	tok     TokenizerWorkspace
	one     [1]string
	oneIDs  [1][]int
	oneOut  [1][]float32
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

// Question prepares a multiple-choice question with 2 to 26 options: it
// tokenizes the prompt and evaluates its prefix once (about 144 KiB of keys
// and values per prefix token in exact mode). It allocates; reuse the result
// for every input.
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
	prefix, err := e.tokens.EncodeInto(text, make([]int, 0, len(text)), &q.tok)
	if err != nil {
		return nil, err
	}
	if q.suffix, err = e.tokens.EncodeInto(questionSuffix, make([]int, 0, len(questionSuffix)), &q.tok); err != nil {
		return nil, err
	}
	if len(prefix)+len(q.suffix) >= maxTokens {
		return nil, fmt.Errorf("qwen3: question uses %d of the %d-token limit", len(prefix)+len(q.suffix), maxTokens)
	}
	if q.kv, err = e.eval.NewPrefixKV(len(prefix)); err != nil {
		return nil, err
	}
	scratch := make([]float32, e.model.cfg.hidden)
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("qwen3: model closed")
	}
	if err := e.eval.HiddenLastExtendInto(q.kv, 0, prefix, scratch, e.ws); err != nil {
		return nil, err
	}
	return q, nil
}

// Choose writes the model's probability for each option, in order, that it
// is the answer for input. Leading and trailing white space in input is
// ignored. Warmed calls allocate nothing.
func (q *Question) Choose(ctx context.Context, input string, probs []float32) error {
	if q == nil {
		return errors.New("qwen3: nil question")
	}
	q.one[0], q.oneOut[0] = input, probs
	err := q.ChooseBatch(ctx, q.one[:], q.oneOut[:])
	q.one[0], q.oneOut[0] = "", nil
	return err
}

// ChooseTokens is Choose for input already tokenized with the model's
// tokenizer; it involves no strings at all.
func (q *Question) ChooseTokens(ctx context.Context, input []int, probs []float32) error {
	if q == nil {
		return errors.New("qwen3: nil question")
	}
	q.oneIDs[0], q.oneOut[0] = input, probs
	err := q.chooseTokens(ctx, q.oneIDs[:], q.oneOut[:])
	q.oneIDs[0], q.oneOut[0] = nil, nil
	return err
}

// ChooseBatch answers the question for every input, writing each input's
// option probabilities to the matching probs row. Inputs share forward
// passes, which is much faster than separate Choose calls. Buffers grow to
// the largest batch seen; later batches up to that size allocate nothing.
func (q *Question) ChooseBatch(ctx context.Context, inputs []string, probs [][]float32) error {
	if q == nil {
		return errors.New("qwen3: nil question")
	}
	if len(probs) != len(inputs) {
		return fmt.Errorf("qwen3: %d probability rows for %d inputs", len(probs), len(inputs))
	}
	need := 0
	for _, in := range inputs {
		need += len(in) + len(q.suffix)
	}
	if need > cap(q.ids) {
		q.ids = make([]int, 0, 2*need)
	}
	q.growBatch(len(inputs))
	ids := q.ids[:0]
	for i, in := range inputs {
		start := len(ids)
		got, err := q.m.tokens.EncodeInto(strings.TrimSpace(in), ids[start:start:cap(ids)], &q.tok)
		if err != nil {
			return fmt.Errorf("qwen3: tokenize input %d: %w", i, err)
		}
		ids = ids[:start+len(got)]
		q.seqs[i] = ids[start:len(ids):len(ids)]
	}
	return q.chooseTokens(ctx, q.seqs[:len(inputs)], probs)
}

func (q *Question) growBatch(n int) {
	if n <= len(q.seqs) {
		return
	}
	n = max(n, 2*len(q.seqs))
	q.seqs = make([][]int, n)
	hidden := make([]float32, n*q.m.model.cfg.hidden)
	q.hidden = make([][]float32, n)
	for i := range q.hidden {
		q.hidden[i] = hidden[i*q.m.model.cfg.hidden : (i+1)*q.m.model.cfg.hidden]
	}
}

// chooseTokens appends the suffix to each input (in place when the input
// lives in q.ids, otherwise into q.ids) and answers all of them.
func (q *Question) chooseTokens(ctx context.Context, inputs [][]int, probs [][]float32) error {
	if len(probs) != len(inputs) {
		return fmt.Errorf("qwen3: %d probability rows for %d inputs", len(probs), len(inputs))
	}
	for i, p := range probs {
		if len(p) != q.options {
			return fmt.Errorf("qwen3: row %d has %d probabilities for %d options", i, len(p), q.options)
		}
	}
	q.growBatch(len(inputs))
	room := maxTokens - len(q.kv.tokens) - len(q.suffix)
	need := 0
	for _, in := range inputs {
		need += min(len(in), room) + len(q.suffix)
	}
	// Assemble input+suffix sequences after a region as long as all inputs,
	// so inputs that ChooseBatch tokenized into the front of q.ids are never
	// overwritten while being copied. Growing leaves old inputs readable.
	used := 0
	for _, in := range inputs {
		used += len(in)
	}
	if used+need > cap(q.ids) {
		q.ids = make([]int, 0, 2*(used+need))
	}
	buf := q.ids[:cap(q.ids)]
	at := used
	for i, in := range inputs {
		if len(in) > room {
			in = in[len(in)-room:] // keep the input's last tokens
		}
		start := at
		at += copy(buf[at:], in)
		at += copy(buf[at:], q.suffix)
		q.seqs[i] = buf[start:at:at]
	}
	seqs := q.seqs[:len(inputs)]
	e := q.m
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("qwen3: model closed")
	}
	for start := 0; start < len(seqs); {
		if err := ctx.Err(); err != nil {
			return err
		}
		end, tokens := start, 0
		for end < len(seqs) && (end == start || tokens+len(seqs[end]) <= batchTokens) {
			tokens += len(seqs[end])
			end++
		}
		if err := e.eval.HiddenLastSharedInto(q.kv, seqs[start:end], q.hidden[start:end], e.ws); err != nil {
			return err
		}
		start = end
	}
	for i := range seqs {
		q.letterProbs(q.hidden[i], probs[i])
	}
	return nil
}

func (q *Question) letterProbs(hidden, probs []float32) {
	rows := q.m.letters.rows
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
}
