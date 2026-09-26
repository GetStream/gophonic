// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"errors"
	"fmt"
)

// Decoder holds the heads and input tables of Evaluator.DecodeInto: at each
// step a head scores the next token from the last state, and a table gives
// the row that token is read back as. A code predictor has a head and a
// table per codebook; a draft model has its one head and embedding table at
// every step.
type Decoder struct {
	e             *Evaluator
	rows          int // tokens a head scores
	heads, tables int
	gpu           *gpuDecoder
}

// Sampling draws a step's token among the TopK likeliest, weighed by
// exp(logit / Temperature), or takes the likeliest when TopK is at most one
// or Temperature is not positive; ties go to the lower token. Draws are
// counter-based: step i of a DecodeInto draws with Seed and Draw+i, so equal
// inputs draw equal tokens.
type Sampling struct {
	TopK        int
	Temperature float32
	Seed, Draw  uint64
}

// NewDecoder prepares heads (each rows×hidden, as a checkpoint stores one)
// and tables (each rows input rows of the hidden width, one per token) for
// DecodeInto, in FP16, with the final norm's weight folded into the heads.
// Only weights on the GPU decode, where a whole run is one submission;
// elsewhere NewDecoder fails and callers step on the CPU.
func (e *Evaluator) NewDecoder(heads, tables [][]float32, rows int) (*Decoder, error) {
	if e == nil || e.m == nil {
		return nil, errors.New("qwen3: nil evaluator")
	}
	c := &e.m.cfg
	switch {
	case e.m.gpu == nil:
		return nil, errors.New("qwen3: decoding runs on the GPU; these weights are not on it")
	case c.hybrid:
		return nil, errors.New("qwen3: a hybrid model does not decode on the GPU")
	case rows <= 0 || len(heads) == 0 || len(heads) > maxDecodeSteps:
		return nil, fmt.Errorf("qwen3: %d heads of %d rows; want 1 to %d heads", len(heads), rows, maxDecodeSteps)
	}
	for _, m := range append(heads[:len(heads):len(heads)], tables...) {
		if len(m) != rows*c.hidden {
			return nil, fmt.Errorf("qwen3: a head or table of %d values; want %d rows of %d", len(m), rows, c.hidden)
		}
	}
	d := &Decoder{e: e, rows: rows, heads: len(heads), tables: len(tables)}
	var err error
	d.gpu, err = e.m.gpu.newDecoder(heads, tables, rows, e.m.finalNorm)
	return d, err
}

// maxDecodeSteps bounds the steps of one DecodeInto.
const maxDecodeSteps = 64

// Close releases the decoder's GPU memory.
func (d *Decoder) Close() error {
	if d != nil {
		if d.gpu != nil {
			d.gpu.release()
		}
		// A closed decoder should not pin its evaluator and the model's
		// weights while the caller keeps the handle around.
		*d = Decoder{}
	}
	return nil
}

// DecodeInto evaluates ids, with embeds, as the continuation of the first
// keep tokens of kv, then draws len(tokens) tokens as s says, all in one
// GPU submission: tokens[i] from head i's logits of the last state, each but
// the last then evaluated as its row of table i. When logits is not nil it
// receives each step's logits, rows per step. Afterwards kv holds the first
// keep tokens, ids, and the tokens evaluated, which it records as embedded
// rows.
func (e *Evaluator) DecodeInto(d *Decoder, kv *PrefixKV, keep int, ids []int, embeds Embeds, s Sampling, tokens []int, logits []float32, ws *Workspace) error {
	switch {
	case d == nil || d.e != e || d.gpu == nil:
		return errors.New("qwen3: the decoder belongs to another evaluator, or is closed")
	case kv == nil || kv.owner != e || kv.gpu == nil:
		return errors.New("qwen3: the prefix store belongs to another evaluator")
	case ws == nil || ws.gpu == nil:
		return errors.New("qwen3: a GPU workspace decodes")
	case keep < 0 || keep > len(kv.tokens):
		return fmt.Errorf("qwen3: keep %d outside the %d stored tokens", keep, len(kv.tokens))
	case len(ids) == 0:
		return errors.New("qwen3: no input tokens")
	case len(tokens) == 0 || len(tokens) > d.heads || len(tokens)-1 > d.tables:
		return fmt.Errorf("qwen3: %d steps for %d heads and %d tables", len(tokens), d.heads, d.tables)
	case keep+len(ids)+len(tokens)-1 > kv.capacity:
		return fmt.Errorf("qwen3: %d tokens exceed prefix capacity %d", keep+len(ids)+len(tokens)-1, kv.capacity)
	case logits != nil && len(logits) != len(tokens)*d.rows:
		return fmt.Errorf("qwen3: %d logits for %d steps of %d", len(logits), len(tokens), d.rows)
	case kv.gpu.recurrent():
		return errors.New("qwen3: a hybrid model does not decode on the GPU")
	}
	n := 0
	for _, id := range ids {
		if len(embeds.Rows) != 0 && id == embeds.Token {
			n++
		}
	}
	if len(embeds.Rows) != n*e.m.cfg.hidden || e.m.embed == nil && n != len(ids) {
		return fmt.Errorf("qwen3: %d embedding values for %d placeholders", len(embeds.Rows), n)
	}
	if s.TopK <= 1 || s.Temperature <= 0 {
		s.TopK, s.Temperature = 1, 1
	}
	s.TopK = min(s.TopK, d.rows)
	kv.tokens = kv.tokens[:keep]
	if err := ws.gpu.decode(e.m, d.gpu, kv.gpu, keep, ids, embeds, s, tokens, logits); err != nil {
		return err
	}
	for _, id := range ids {
		if len(embeds.Rows) != 0 && id == embeds.Token {
			id = -1
		}
		kv.tokens = append(kv.tokens, id)
	}
	for range tokens[1:] {
		kv.tokens = append(kv.tokens, -1)
	}
	return nil
}
