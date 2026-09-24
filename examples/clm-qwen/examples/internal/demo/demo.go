// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package demo loads the local Qwen3-8B encoder and CLM head once for the
// task examples and scores texts with one reusable workspace.
package demo

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/GetStream/gophonic/clm"
	clmqwen "github.com/GetStream/gophonic/examples/clm-qwen"
)

const maxCandidates = 64

// Model is a loaded Qwen3-8B encoder and CLM head. It is not safe for
// concurrent use.
type Model struct {
	enc    *clmqwen.Encoder
	engine *clm.Engine
	ws     *clm.RankWorkspace
	ranked []clm.RankedCandidate
	probs  []float32
	calls  int
	busy   time.Duration
}

// Open parses the command line and loads the models. -qwen and -head default
// to $GOPHONIC_QWEN3_MODEL and $GOPHONIC_CLM_HEAD_BUNDLE.
func Open() (*Model, error) {
	qwen := flag.String("qwen", os.Getenv("GOPHONIC_QWEN3_MODEL"), "Qwen3-8B safetensors directory")
	head := flag.String("head", os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE"), "converted CLM .gclm head bundle")
	weights := flag.String("weights", "", "f16 (exact, default) or int8 (half the memory)")
	flag.Parse()
	if *qwen == "" || *head == "" {
		return nil, errors.New("set -qwen and -head, or GOPHONIC_QWEN3_MODEL and GOPHONIC_CLM_HEAD_BUNDLE")
	}
	start := time.Now()
	h, err := clm.Load(*head)
	if err != nil {
		return nil, fmt.Errorf("load CLM head: %w", err)
	}
	enc, err := clmqwen.OpenWithOptions(*qwen, clmqwen.Options{Weights: *weights})
	if err != nil {
		return nil, fmt.Errorf("load Qwen3-8B: %w", err)
	}
	m := &Model{enc: enc, ranked: make([]clm.RankedCandidate, maxCandidates), probs: make([]float32, maxCandidates)}
	if m.engine, err = clm.NewEngine(h, enc); err == nil {
		m.ws, err = m.engine.NewWorkspace(maxCandidates)
	}
	if err != nil {
		enc.Close()
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "loaded Qwen3-8B and CLM head in %.1fs\n", time.Since(start).Seconds())
	return m, nil
}

// Close stops the encoder's worker threads.
func (m *Model) Close() error { return m.enc.Close() }

// Probs scores each candidate as a response to state and returns softmax
// probabilities in candidate order. The slice is overwritten by the next call.
func (m *Model) Probs(state string, candidates []string) ([]float32, error) {
	n := len(candidates)
	if n > maxCandidates {
		return nil, fmt.Errorf("demo: %d candidates, limit is %d", n, maxCandidates)
	}
	start := time.Now()
	if err := m.engine.RankInto(context.Background(), state, candidates, 1, m.ranked[:n], m.ws); err != nil {
		return nil, err
	}
	m.calls++
	m.busy += time.Since(start)
	for _, r := range m.ranked[:n] {
		for i, c := range candidates {
			if c == r.Candidate {
				m.probs[i] = r.Probability
			}
		}
	}
	return m.probs[:n], nil
}

// Best returns the index of the largest probability.
func Best(probs []float32) int {
	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	return best
}

// Summary reports scoring calls, mean latency, and embedding-cache hits.
// Repeated candidate texts are served from the encoder's exact cache, so
// after the first call each one costs about one short-text inference.
func (m *Model) Summary() string {
	hits, lookups := m.enc.CacheStats()
	mean := time.Duration(0)
	if m.calls > 0 {
		mean = m.busy / time.Duration(m.calls)
	}
	return fmt.Sprintf("%d texts scored in %v (%v each); embedding cache %d/%d hits",
		m.calls, m.busy.Round(time.Millisecond), mean.Round(time.Millisecond), hits, lookups)
}
