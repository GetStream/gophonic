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
	"strings"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

// moePrompt is the reference prompt of the mixture-of-experts tests.
const moePrompt = "The capital of France is"

type moeModel struct {
	tokens  *Tokenizer
	weights *Weights
	eval    *Evaluator
	ws      *Workspace
}

func loadMoE(tb testing.TB) moeModel {
	path := testmodels.Path(tb, testmodels.Qwen3MoE)
	tokens, err := LoadTokenizer(path)
	if err != nil {
		tb.Fatal(err)
	}
	weights, err := Load(path, LoadOptions{Head: "lm_head.weight"})
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { weights.Release() })
	eval, err := NewEvaluator(weights)
	if err != nil {
		tb.Fatal(err)
	}
	ws, err := eval.NewWorkspace(4)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { ws.Close() })
	return moeModel{tokens, weights, eval, ws}
}

// moeIDs returns n token ids of ordinary text.
func moeIDs(tb testing.TB, m moeModel, n int) []int {
	var tws TokenizerWorkspace
	text := strings.Repeat("The river bends past the old mill, where the miller's daughter counts the sacks of grain and sings to the geese. ", 1+n/20)
	ids, err := m.tokens.EncodeInto(text, make([]int, 0, 2*len(text)), &tws)
	if err != nil {
		tb.Fatal(err)
	}
	return ids[:n]
}

// Qwen3-30B-A3B on the GPU matches the float32 reference forward pass of
// its official checkpoint, in batches and a token at a time.
func TestMoEMatchesReference(t *testing.T) {
	m := loadMoE(t)
	tokens, eval, ws := m.tokens, m.eval, m.ws
	var tws TokenizerWorkspace
	ids, err := tokens.EncodeInto(moePrompt, make([]int, 0, 32), &tws)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("ids %v", ids)
	c := m.weights.Config()
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
	t.Logf("batch vs token by token: cosine %.6f", cosine(batch, token))
	logits := make([]float32, c.Vocab)
	if err := eval.LogitsInto(batch, logits, ws); err != nil {
		t.Fatal(err)
	}
	top := topIDs(logits, 5)
	t.Logf("top %v %q", top, tokens.DecodeAppend(nil, top[:1], true))
	// Greedy continuation, for the eye.
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
	raw, err := os.ReadFile(filepath.Join(testmodels.Dir(), testmodels.Qwen3MoEReference))
	if err != nil {
		t.Skipf("no reference state (run internal/qwen3lm/tools/moe_reference.py with ids %v): %v", ids, err)
	}
	ref := make([]float32, len(raw)/4)
	for i := range ref {
		ref[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	cb, ct := cosine(batch, ref), cosine(token, ref)
	t.Logf("cosine to the reference: batch %.6f, token by token %.6f", cb, ct)
	if cb < 0.99 || ct < 0.99 {
		t.Fatalf("the MoE state strays from the reference: cosine %.4f batch, %.4f token by token", cb, ct)
	}
}

// Batches that group their rows by expert, in 16- and 32-pair tiles, are as
// accurate as rows that stream their own experts: nearly equal on short
// inputs, and as close to token-by-token decoding on long ones, where
// routing amplifies any rounding (both reach a cosine of about 0.997).
func TestMoEGroupedExperts(t *testing.T) {
	m := loadMoE(t)
	c := m.weights.Config()
	grouped, rows, token := make([]float32, c.Hidden), make([]float32, c.Hidden), make([]float32, c.Hidden)
	defer func(n int) { moeGroupRows = n }(moeGroupRows)
	for _, n := range []int{4, 17, 100, 700} {
		ids := moeIDs(t, m, n)
		moeGroupRows = 4
		if err := m.eval.HiddenLastInto(ids, grouped, m.ws); err != nil {
			t.Fatal(err)
		}
		moeGroupRows = math.MaxInt
		if err := m.eval.HiddenLastInto(ids, rows, m.ws); err != nil {
			t.Fatal(err)
		}
		cos := cosine(grouped, rows)
		t.Logf("%d tokens: cosine %.7f", n, cos)
		if n < 700 {
			if cos < 0.9998 {
				t.Errorf("%d tokens: grouped experts diverge: cosine %.7f", n, cos)
			}
			continue
		}
		gpuTokenByToken = true
		err := m.eval.HiddenLastInto(ids, token, m.ws)
		gpuTokenByToken = false
		if err != nil {
			t.Fatal(err)
		}
		cg, cr := cosine(grouped, token), cosine(rows, token)
		t.Logf("%d tokens, to token by token: grouped %.7f, rows %.7f", n, cg, cr)
		if cg < cr-0.001 {
			t.Errorf("%d tokens: grouping the experts costs accuracy: %.5f to token by token, %.5f by rows", n, cg, cr)
		}
	}
}

// BenchmarkMoEPass times forward passes of the mixture of experts, with
// experts grouped and a row at a time.
func BenchmarkMoEPass(b *testing.B) {
	m := loadMoE(b)
	dst := make([]float32, m.weights.Config().Hidden)
	defer func(n int) { moeGroupRows = n }(moeGroupRows)
	for _, n := range []int{1, 8, 16, 32, 64, 256, 1024} {
		ids := moeIDs(b, m, n)
		for _, mode := range []struct {
			name string
			from int
		}{{"grouped", 2}, {"rows", math.MaxInt}} {
			if n == 1 && mode.from == 2 {
				continue
			}
			b.Run(fmt.Sprintf("%d/%s", n, mode.name), func(b *testing.B) {
				moeGroupRows = mode.from
				for b.Loop() {
					if err := m.eval.HiddenLastInto(ids, dst, m.ws); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func topIDs(logits []float32, k int) []int {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	slices.SortFunc(idx, func(a, b int) int {
		switch {
		case logits[a] > logits[b]:
			return -1
		case logits[a] < logits[b]:
			return 1
		}
		return 0
	})
	return idx[:k]
}
