// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"fmt"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
	"github.com/GetStream/gophonic/internal/testmodels"
)

// officialLM is the official Qwen3-8B checkpoint loaded for low-level tests.
type officialLM struct {
	weights *Weights
	eval    *Evaluator
	ws      *Workspace
	tokens  *Tokenizer
	tokenWS TokenizerWorkspace
	hidden  int
}

func loadOfficialLM(t testing.TB, format string) *officialLM {
	t.Helper()
	path := testmodels.Path(t, testmodels.Qwen3)
	tokens, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := LoadWeights(path, format)
	if err != nil {
		t.Fatal(err)
	}
	eval, err := NewEvaluator(weights)
	if err != nil {
		t.Fatal(err)
	}
	ws, err := eval.NewWorkspace(max(1, min(PerformanceCores(), 16)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = ws.Close()
		weights.Release()
	})
	return &officialLM{weights: weights, eval: eval, ws: ws, tokens: tokens, hidden: weights.Config().Hidden}
}

func (lm *officialLM) rows(n int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, lm.hidden)
	}
	return out
}

// TestOfficialGPUBatchMatchesTokens compares the batched GPU forward pass with
// the token-by-token one on a real checkpoint.
func TestOfficialGPUBatchMatchesTokens(t *testing.T) {
	for _, format := range []string{WeightsInt8, WeightsGPU, WeightsGPUQ4} {
		lm := loadOfficialLM(t, format)
		ids, err := lm.tokens.EncodeInto("The quick brown fox jumps over the lazy dog while the band plays a slow song about rivers and mountains far away.", make([]int, 0, 256), &lm.tokenWS)
		if err != nil {
			t.Fatal(err)
		}
		out := lm.rows(2)
		batch, tokens := out[0], out[1]
		if err := lm.eval.HiddenLastInto(ids, batch, lm.ws); err != nil {
			t.Fatal(err)
		}
		gpuTokenByToken = true
		err = lm.eval.HiddenLastInto(ids, tokens, lm.ws)
		gpuTokenByToken = false
		if err != nil {
			t.Fatal(err)
		}
		cos, maxAbs := lmtest.VectorParity(batch, tokens)
		t.Logf("%s: %d tokens, batched vs token-by-token cosine %.7f (max_abs %.3g)", format, len(ids), cos, maxAbs)
		if cos < 0.99999 {
			t.Errorf("%s: batched path diverges: cosine %.7f", format, cos)
		}
		// Packing several sequences into one pass matches evaluating each.
		seqs := [][]int{ids[3:9], ids, ids[:1]}
		packed := lm.rows(len(seqs))
		if err := lm.eval.HiddenLastBatchInto(seqs, packed, lm.ws); err != nil {
			t.Fatal(err)
		}
		for s, seq := range seqs {
			alone := lm.rows(1)[0]
			if err := lm.eval.HiddenLastInto(seq, alone, lm.ws); err != nil {
				t.Fatal(err)
			}
			if cos, _ := lmtest.VectorParity(packed[s], alone); cos < 0.99999 {
				t.Errorf("%s: packed sequence %d diverges: cosine %.7f", format, s, cos)
			}
		}
	}
}

// BenchmarkOfficialGPUScaling times one GPU forward pass per input length.
func BenchmarkOfficialGPUScaling(b *testing.B) {
	lm := loadOfficialLM(b, WeightsGPU)
	dst := lm.rows(1)[0]
	for _, n := range []int{1, 12, 16, 17, 32, 64, 128, 192} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = 1000 + i*37
		}
		b.Run(fmt.Sprintf("%dtok", n), func(b *testing.B) {
			for b.Loop() {
				if err := lm.eval.HiddenLastInto(ids, dst, lm.ws); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestOfficialGPUFlashAttention compares the tiled attention kernel with the
// per-key loop on packed sequences, a long input, and a shared prefix.
func TestOfficialGPUFlashAttention(t *testing.T) {
	lm := loadOfficialLM(t, WeightsGPU)
	long := make([]int, 150)
	for i := range long {
		long[i] = 1000 + (i*7919)%50000
	}
	seqs := [][]int{long[:5], long, long[20:31]}
	compare := func(what string, run func() [][]float32) {
		gpuScalarAttention = false
		flash := run()
		gpuScalarAttention = true
		scalar := run()
		gpuScalarAttention = false
		for s := range flash {
			cos, maxAbs := lmtest.VectorParity(flash[s], scalar[s])
			t.Logf("%s %d: cosine %.7f max_abs %.3g", what, s, cos, maxAbs)
			if cos < 0.99999 {
				t.Errorf("%s %d: flash attention diverges", what, s)
			}
		}
	}
	compare("packed sequence", func() [][]float32 {
		out := lm.rows(len(seqs))
		if err := lm.eval.HiddenLastBatchInto(seqs, out, lm.ws); err != nil {
			t.Fatal(err)
		}
		return out
	})
	// Continuations of one shared prefix, as Question.ChooseBatch runs them.
	kv, err := lm.eval.NewPrefixKV(512)
	if err != nil {
		t.Fatal(err)
	}
	if err := lm.eval.HiddenLastExtendInto(kv, 0, long[:90], lm.rows(1)[0], lm.ws); err != nil {
		t.Fatal(err)
	}
	compare("shared-prefix continuation", func() [][]float32 {
		out := lm.rows(len(seqs))
		if err := lm.eval.HiddenLastSharedInto(kv, seqs, out, lm.ws); err != nil {
			t.Fatal(err)
		}
		return out
	})
}
