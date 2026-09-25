// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"math/rand"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

// checkTail extends kv, after its first keep tokens, with ids in one pass
// that returns the last k states, and compares each state and its logits
// with those of the same positions evaluated one call at a time.
func checkTail(t *testing.T, e *Evaluator, ws *Workspace, keep int, ids []int, embeds Embeds, k int, gate float64) {
	t.Helper()
	c := e.m.cfg
	kv, err := e.NewPrefixKV(keep + len(ids) + 1)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := e.NewPrefixKV(keep + len(ids) + 1)
	if err != nil {
		t.Fatal(err)
	}
	tail := make([]float32, k*c.hidden)
	if err := e.HiddenTailExtendEmbedInto(kv, 0, ids, embeds, tail, ws); err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, k*c.vocab)
	if err := e.LogitsRowsInto(tail, logits, ws); err != nil {
		t.Fatal(err)
	}
	// The same positions one at a time: the head up to the tail in one
	// pass, then a token per call.
	one, oneLogits := make([]float32, c.hidden), make([]float32, c.vocab)
	head := len(ids) - k + 1
	if err := e.HiddenLastExtendEmbedInto(ref, 0, ids[:head], embeds, one, ws); err != nil {
		t.Fatal(err)
	}
	for i := range k {
		if i > 0 {
			if err := e.HiddenLastExtendInto(ref, head+i-1, ids[head+i-1:head+i], one, ws); err != nil {
				t.Fatal(err)
			}
		}
		if cos, maxAbs := lmtest.VectorParity(tail[i*c.hidden:(i+1)*c.hidden], one); cos < gate {
			t.Fatalf("tail state %d of %d: cosine %.9f max_abs %g", i, k, cos, maxAbs)
		}
		if err := e.LogitsInto(one, oneLogits, ws); err != nil {
			t.Fatal(err)
		}
		if cos, maxAbs := lmtest.VectorParity(logits[i*c.vocab:(i+1)*c.vocab], oneLogits); cos < gate {
			t.Fatalf("tail logits %d of %d: cosine %.9f max_abs %g", i, k, cos, maxAbs)
		}
	}
	if len(kv.Tokens()) != len(ids) {
		t.Fatalf("stored %d tokens, want %d", len(kv.Tokens()), len(ids))
	}
}

// A tail extension returns the states of its last positions, each equal to
// evaluating that position on its own, with embedded rows spliced in.
func TestTailExtensionMatchesPositions(t *testing.T) {
	ck := lmtest.Write(t, 12)
	s := lmtest.Shape
	const placeholder = 5
	rng := rand.New(rand.NewSource(4))
	rows := make([]float32, 3*s.Hidden)
	for i := range rows {
		rows[i] = float32(rng.NormFloat64())
	}
	ids := []int{1, 2, placeholder, placeholder, placeholder, 4, 7, 9, 3, 8, 6, 2, 1}
	for _, format := range []string{WeightsF16, WeightsInt8} {
		m, err := Load(ck.Dir, LoadOptions{Format: format, Head: "lm_head.weight"})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := NewEvaluator(m)
		ws, _ := e.NewWorkspace(3)
		for _, k := range []int{1, 2, 5, 8} {
			checkTail(t, e, ws, 0, ids, Embeds{placeholder, rows}, k, 0.99999)
		}
		ws.Close()
	}
}
