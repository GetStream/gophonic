// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

// Qwen3.6-35B-A3B on the GPU, Gated DeltaNet and gated attention layers with
// experts, matches the float32 reference forward pass of its official
// checkpoint, in batches and a token at a time.
func TestQwen36MatchesReference(t *testing.T) {
	path := testmodels.Path(t, testmodels.Qwen36)
	tokens, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	var tws TokenizerWorkspace
	ids, err := tokens.EncodeInto(moePrompt, make([]int, 0, 32), &tws)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := Load(path, LoadOptions{Head: "lm_head.weight"})
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Release()
	eval, err := NewEvaluator(weights)
	if err != nil {
		t.Fatal(err)
	}
	ws, err := eval.NewWorkspace(4)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	c := weights.Config()
	batch, token := make([]float32, c.Hidden), make([]float32, c.Hidden)
	if err := eval.HiddenLastInto(ids, batch, ws); err != nil {
		t.Fatal(err)
	}
	gpuTokenByToken = true
	err = eval.HiddenLastInto(ids, token, ws)
	gpuTokenByToken = false
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("ids %v; batch vs token by token: cosine %.6f", ids, cosine(batch, token))
	logits := make([]float32, c.Vocab)
	if err := eval.LogitsInto(batch, logits, ws); err != nil {
		t.Fatal(err)
	}
	top := topIDs(logits, 5)
	t.Logf("top %v %q", top, tokens.DecodeAppend(nil, top[:1], true))
	seq := slices.Clone(ids)
	for range 12 {
		if err := eval.HiddenLastInto(seq, batch, ws); err != nil {
			t.Fatal(err)
		}
		if err := eval.LogitsInto(batch, logits, ws); err != nil {
			t.Fatal(err)
		}
		seq = append(seq, topIDs(logits, 1)[0])
	}
	t.Logf("greedy: %q", tokens.DecodeAppend(nil, seq, true))
	if err := eval.HiddenLastInto(ids, batch, ws); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(testmodels.Dir(), testmodels.Qwen36Reference))
	if err != nil {
		t.Skipf("no reference state (run internal/qwen3lm/tools/qwen35_reference.py with ids %v): %v", ids, err)
	}
	ref := make([]float32, len(raw)/4)
	for i := range ref {
		ref[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	cb, ct := cosine(batch, ref), cosine(token, ref)
	t.Logf("cosine to the reference: batch %.6f, token by token %.6f", cb, ct)
	if cb < 0.99 || ct < 0.99 {
		t.Fatalf("the Qwen3.6 state strays from the reference: cosine %.4f batch, %.4f token by token", cb, ct)
	}
}

// A conversation on Qwen3.6 rewinds: extending a prefix from inside what it
// already evaluated (a dropped speculative message, a cut reply) restores
// the recurrent state from a snapshot and computes exactly what extending
// without the detour computes; continuations of a shared prefix compute what
// extending the prefix does. Against evaluating from scratch, which splits
// its work differently, the states agree only as well as the experts'
// routing allows (a cosine of about 0.998 on repeated text).
func TestQwen36Rewind(t *testing.T) {
	path := testmodels.Path(t, testmodels.Qwen36)
	weights, err := Load(path, LoadOptions{Head: "lm_head.weight"})
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Release()
	m := moeModel{weights: weights}
	if m.tokens, err = LoadTokenizer(path); err != nil {
		t.Fatal(err)
	}
	if m.eval, err = NewEvaluator(weights); err != nil {
		t.Fatal(err)
	}
	if m.ws, err = m.eval.NewWorkspace(4); err != nil {
		t.Fatal(err)
	}
	defer m.ws.Close()
	h := weights.Config().Hidden
	ids := moeIDs(t, m, 90)
	a, b1, b2 := ids[:40], ids[40:52], ids[60:75]
	newKV := func() *PrefixKV {
		kv, err := m.eval.NewPrefixKV(256)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { kv.Close() })
		return kv
	}
	extend := func(kv *PrefixKV, keep int, seq []int, dst []float32) {
		t.Helper()
		if err := m.eval.HiddenLastExtendInto(kv, keep, seq, dst, m.ws); err != nil {
			t.Fatal(err)
		}
	}
	fresh := make([]float32, h)
	compare := func(what string, got, direct []float32, tokens []int) {
		t.Helper()
		if err := m.eval.HiddenLastInto(tokens, fresh, m.ws); err != nil {
			t.Fatal(err)
		}
		exact, scratch := cosine(got, direct), cosine(got, fresh)
		t.Logf("%s: cosine %.7f to the direct path, %.5f from scratch", what, exact, scratch)
		if exact < 0.9999999 || scratch < 0.99 {
			t.Errorf("%s: the rewound state diverges", what)
		}
	}
	// One conversation takes detours; another goes straight.
	kv, direct := newKV(), newKV()
	got, want := make([]float32, h), make([]float32, h)
	extend(kv, 0, a, got)
	extend(kv, len(a), b1, got)
	extend(kv, len(a)+4, b2, got) // from inside b1: the snapshot at len(a), then 4 tokens again
	extend(direct, 0, a, want)
	extend(direct, len(a), b1[:4], want)
	extend(direct, len(a)+4, b2, want)
	compare("rewound into a message", got, want, slices.Concat(a, b1[:4], b2))
	for i, id := range b1[:3] { // a token at a time, as a reply decodes
		extend(kv, len(a)+4+len(b2)+i, []int{id}, got)
		extend(direct, len(a)+4+len(b2)+i, []int{id}, want)
	}
	compare("decoded", got, want, slices.Concat(a, b1[:4], b2, b1[:3]))
	extend(kv, 7, b1, got) // before every snapshot but the first
	restart := newKV()
	extend(restart, 0, a[:7], want)
	extend(restart, 7, b1, want)
	compare("rewound to the start", got, want, slices.Concat(a[:7], b1))
	// Continuations of a shared prefix each start from its state.
	seqs, outs := [][]int{b1[:5], b2[:9]}, [][]float32{make([]float32, h), make([]float32, h)}
	if err := m.eval.HiddenLastSharedInto(kv, seqs, outs, m.ws); err != nil {
		t.Fatal(err)
	}
	prefix := slices.Clone(kv.Tokens()) // restart holds the same tokens
	for i, s := range seqs {
		extend(restart, len(prefix), s, want)
		compare(fmt.Sprintf("shared continuation %d", i), outs[i], want, slices.Concat(prefix, s))
	}
}

// BenchmarkHybridDecode times what a chat does with a model: prefill a
// 256-token prompt, then decode a token at a time. MODEL picks
// testmodels.Qwen36 (default) or another snapshot for comparison.
func BenchmarkHybridDecode(b *testing.B) {
	name := os.Getenv("MODEL")
	if name == "" {
		name = testmodels.Qwen36
	}
	path := testmodels.Path(b, name)
	weights, err := Load(path, LoadOptions{Head: "lm_head.weight"})
	if err != nil {
		b.Fatal(err)
	}
	defer weights.Release()
	m := moeModel{weights: weights}
	m.tokens, _ = LoadTokenizer(path)
	m.eval, _ = NewEvaluator(weights)
	m.ws, _ = m.eval.NewWorkspace(4)
	defer m.ws.Close()
	h := weights.Config().Hidden
	ids := moeIDs(b, m, 256+64)
	kv, err := m.eval.NewPrefixKV(512)
	if err != nil {
		b.Fatal(err)
	}
	defer kv.Close()
	dst := make([]float32, h)
	b.Run("prefill256", func(b *testing.B) {
		for b.Loop() {
			if err := m.eval.HiddenLastExtendInto(kv, 0, ids[:256], dst, m.ws); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("decode", func(b *testing.B) {
		i := 0
		for b.Loop() {
			n := 256 + i%64
			if err := m.eval.HiddenLastExtendInto(kv, n, ids[n:n+1], dst, m.ws); err != nil {
				b.Fatal(err)
			}
			i++
			if i%64 == 0 {
				b.StopTimer()
				if err := m.eval.HiddenLastExtendInto(kv, 0, ids[:256], dst, m.ws); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
			}
		}
	})
}
