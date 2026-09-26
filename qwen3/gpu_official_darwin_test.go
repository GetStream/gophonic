// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3

import (
	"context"
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
	"github.com/GetStream/gophonic/internal/testmodels"
)

// TestOfficialGPUFidelity compares GPU formats with the exact CPU path on
// varied texts: embedding cosine and Choose probabilities.
func TestOfficialGPUFidelity(t *testing.T) {
	if testing.Short() {
		t.Skip("loads every weight format; skipped with -short")
	}
	path := testmodels.Path(t, testmodels.Qwen3)
	texts := []string{
		"hello",
		"The quarterly report shows revenue grew eight percent while costs stayed flat.",
		"i think the meeting got moved to thursday but nobody told me",
		"func (s *Server) Close() error { return s.ln.Close() }",
		"Can I return a jacket I bought online if I already removed the tags?",
		"La reunión se canceló por la lluvia, pero la haremos el lunes.",
		"The patient reported mild headaches and dizziness after starting the new medication.",
		"Wow. Just wow. Best customer service I've ever had, they fixed it in five minutes.",
		"Explain why the sky is blue in one sentence.",
		"User: my order never arrived and the tracking link is broken\nAgent: I'm sorry, let me check that for you.",
	}
	question := "What does the writer want?"
	options := []string{"a refund", "information", "to complain", "to praise", "nothing"}
	run := func(format string) ([][]float32, [][]float32) {
		m, err := Open(path, Options{Format: format, CacheEntries: -1, PrefixCacheTokens: -1})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		emb := make([][]float32, len(texts))
		for i := range emb {
			emb[i] = make([]float32, m.Width())
		}
		if err := m.Embed(context.Background(), texts, emb); err != nil {
			t.Fatal(err)
		}
		q, err := m.Question(question, options)
		if err != nil {
			t.Fatal(err)
		}
		probs := make([][]float32, len(texts))
		for i := range probs {
			probs[i] = make([]float32, len(options))
		}
		if err := q.ChooseBatch(context.Background(), texts, probs); err != nil {
			t.Fatal(err)
		}
		return emb, probs
	}
	refEmb, refProbs := run(qwen3lm.WeightsF16)
	for _, format := range []string{qwen3lm.WeightsInt8, qwen3lm.WeightsGPU, qwen3lm.WeightsGPUQ4} {
		emb, probs := run(format)
		minCos, sumCos, maxDP, agree := 1.0, 0.0, 0.0, 0
		for i := range texts {
			cos, _ := lmtest.VectorParity(emb[i], refEmb[i])
			minCos, sumCos = min(minCos, cos), sumCos+cos
			best, refBest := 0, 0
			for j := range options {
				maxDP = max(maxDP, math.Abs(float64(probs[i][j]-refProbs[i][j])))
				if probs[i][j] > probs[i][best] {
					best = j
				}
				if refProbs[i][j] > refProbs[i][refBest] {
					refBest = j
				}
			}
			if best == refBest {
				agree++
			}
		}
		for i := range texts {
			t.Logf("  %-8s %q exact %.3f got %.3f", format, texts[i][:min(24, len(texts[i]))], refProbs[i], probs[i])
		}
		t.Logf("%s vs exact CPU: embedding cosine min %.6f mean %.6f; Choose max|Δp| %.4f, argmax %d/%d",
			format, minCos, sumCos/float64(len(texts)), maxDP, agree, len(texts))
	}
}
