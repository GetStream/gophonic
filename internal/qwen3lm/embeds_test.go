// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

// A decoder nested in a larger checkpoint, with its head, spliced input
// embeddings, and one-token decode steps, matches the float64 reference.
func TestNestedDecoderWithHeadAndEmbedsMatchesReference(t *testing.T) {
	ck := lmtest.WriteNamed(t, 11, func(name string) string {
		if rest, ok := strings.CutPrefix(name, "model."); ok {
			return "thinker.model." + rest
		}
		return "thinker." + name
	})
	s := lmtest.Shape
	headChunkRows = 8 // pack the 23-row head from three chunks
	defer func() { headChunkRows = 0 }()
	const placeholder = 5
	rng := rand.New(rand.NewSource(3))
	rows := make([]float32, 3*s.Hidden)
	for i := range rows {
		rows[i] = float32(rng.NormFloat64())
	}
	ids := []int{1, 2, 3, placeholder, placeholder, placeholder, 4, 7, 9}
	for _, format := range []string{WeightsF16, WeightsInt8} {
		cfg := &TextConfig{ModelType: "qwen3", HiddenSize: s.Hidden, Layers: s.Layers, Heads: s.Heads, KVHeads: s.KVHeads,
			HeadDim: s.HeadDim, Intermediate: s.Inter, Vocab: s.Vocab, MaxPositions: s.MaxPos, RMSNormEps: 1e-6, RopeTheta: 1e6}
		m, err := Load(ck.Dir, LoadOptions{Format: format, Prefix: "thinker.model.", Config: cfg, Head: "thinker.lm_head.weight"})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := NewEvaluator(m)
		ws, _ := e.NewWorkspace(3)
		kv, _ := e.NewPrefixKV(64)
		gate := 0.99999
		if format == WeightsInt8 {
			gate = 0.999
		}
		hidden, logits := make([]float32, s.Hidden), make([]float32, s.Vocab)
		check := func(name string, full []int) {
			t.Helper()
			ref := ck.ReferenceHiddenEmbeds(full, placeholder, rows)
			if cos, _ := lmtest.VectorParity(hidden, ref); cos < gate {
				t.Fatalf("%s %s: hidden cosine %.9f", format, name, cos)
			}
			if err := e.LogitsInto(hidden, logits, ws); err != nil {
				t.Fatal(err)
			}
			want := ck.ReferenceLogits(hidden)
			if cos, _ := lmtest.VectorParity(logits, want); cos < gate {
				t.Fatalf("%s %s: logits cosine %.9f", format, name, cos)
			}
		}
		if err := e.HiddenLastExtendEmbedInto(kv, 0, ids, Embeds{placeholder, rows}, hidden, ws); err != nil {
			t.Fatal(err)
		}
		check("prompt", ids)
		want := []int{1, 2, 3, -1, -1, -1, 4, 7, 9}
		if !slices.Equal(kv.Tokens(), want) {
			t.Fatalf("stored tokens %v, want %v", kv.Tokens(), want)
		}
		if n := kv.CommonPrefix(ids); n != 3 {
			t.Fatalf("common prefix %d across spliced rows, want 3", n)
		}
		full := slices.Clone(ids)
		step := []int{0}
		for _, next := range []int{11, 2, 22} {
			step[0] = next
			if err := e.HiddenLastExtendInto(kv, len(kv.Tokens()), step, hidden, ws); err != nil {
				t.Fatal(err)
			}
			full = append(full, next)
			check("decode step", full)
		}
		// Warm decode steps and logits allocate nothing.
		keep := len(kv.Tokens()) - 1
		if n := testing.AllocsPerRun(10, func() {
			if err := e.HiddenLastExtendInto(kv, keep, step, hidden, ws); err != nil {
				panic(err)
			}
			if err := e.LogitsInto(hidden, logits, ws); err != nil {
				panic(err)
			}
		}); n != 0 {
			t.Fatalf("%s: warm decode step allocated %.1f times", format, n)
		}
		if err := e.HiddenLastExtendEmbedInto(kv, 0, ids, Embeds{placeholder, rows[:s.Hidden]}, hidden, ws); err == nil {
			t.Fatal("accepted fewer embedding rows than placeholders")
		}
		_ = ws.Close()
	}
	m, err := Load(ck.Dir, LoadOptions{Prefix: "thinker.model."})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(1)
	if err := e.LogitsInto(make([]float32, s.Hidden), make([]float32, s.Vocab), ws); err == nil {
		t.Fatal("computed logits without a loaded head")
	}
}
