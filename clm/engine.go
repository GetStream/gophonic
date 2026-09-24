// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clm

import (
	"context"
	"fmt"
	"math"
)

const maxWorkspaceBytes = 256 << 20

// Role identifies the text side being embedded. CLM uses distinct state and
// action projection heads; a local encoder may also use this value to apply
// side-specific prompt handling.
type Role uint8

const (
	StateRole Role = iota + 1
	ActionRole
)

// Embedder supplies raw, post-final-normalization encoder vectors. The output
// buffers are caller-owned and have one 4096-value slice per input. The
// callback is made once per role and batch, never once per vector or element.
// The engine applies CLM's input L2 normalization before the projection heads.
type Embedder interface {
	Embed(ctx context.Context, role Role, texts []string, dst [][]float32) error
}

// EmbedFunc adapts a function to Embedder.
type EmbedFunc func(context.Context, Role, []string, [][]float32) error

func (f EmbedFunc) Embed(ctx context.Context, role Role, texts []string, dst [][]float32) error {
	return f(ctx, role, texts, dst)
}

// Engine combines a trained head pair with a text embedder. Treat it as
// immutable after construction and provide one RankWorkspace per concurrent
// lane. The backbone is deliberately separate: CLM-v0.1-8B was trained against
// Qwen3-8B last-token pooled embeddings, so using a different encoder changes
// the model.
type Engine struct {
	Head     *HeadPair
	Embedder Embedder
}

// NewEngine constructs a text ranker over the supplied state/action heads and
// compatible encoder. Compatibility with the checkpoint's training encoder
// is the caller's responsibility.
func NewEngine(head *HeadPair, embedder Embedder) (*Engine, error) {
	if head == nil {
		return nil, fmt.Errorf("clm: nil head pair")
	}
	if embedder == nil {
		return nil, fmt.Errorf("clm: nil embedder")
	}
	return &Engine{Head: head, Embedder: embedder}, nil
}

// RankedCandidate is one candidate and its softmax probability, sorted best
// first. Rank is one-based.
type RankedCandidate struct {
	Rank        int     `json:"rank"`
	Candidate   string  `json:"candidate"`
	Probability float32 `json:"prob"`
}

// NewWorkspace allocates reusable storage for up to candidateCapacity texts.
// Callers should keep one workspace per concurrent request lane. A warmed
// RankInto call does not allocate in the engine or head code; allocations made
// by an Embedder are specific to that implementation.
func (e *Engine) NewWorkspace(candidateCapacity int) (*RankWorkspace, error) {
	if e == nil || e.Head == nil || e.Embedder == nil {
		return nil, fmt.Errorf("clm: engine is not initialized")
	}
	if candidateCapacity < 1 {
		return nil, fmt.Errorf("clm: candidate capacity must be positive")
	}
	cfg := e.Head.config
	// Bound reusable action vectors and per-candidate metadata before
	// allocating. The extra 64 bytes reserves room for each vector-slice header
	// and score, with margin for architecture differences.
	bytesPerCandidate := 4*cfg.EncoderDim + 64
	maxInt := int(^uint(0) >> 1)
	if candidateCapacity > maxInt/bytesPerCandidate {
		return nil, fmt.Errorf("clm: candidate capacity overflows workspace size")
	}
	if candidateCapacity > maxWorkspaceBytes/bytesPerCandidate {
		return nil, fmt.Errorf("clm: candidate capacity %d exceeds the %d MiB workspace limit", candidateCapacity, maxWorkspaceBytes>>20)
	}
	ws := &RankWorkspace{
		engine:              e,
		candidateCapacity:   candidateCapacity,
		stateEmbedding:      make([]float32, cfg.EncoderDim),
		actionEmbeddingData: make([]float32, candidateCapacity*cfg.EncoderDim),
		actionEmbeddings:    make([][]float32, candidateCapacity),
		scores:              make([]float32, candidateCapacity),
		head:                e.Head.NewWorkspace(),
	}
	ws.stateTexts[0] = ""
	ws.stateOutputs[0] = ws.stateEmbedding
	for i := range ws.actionEmbeddings {
		start := i * cfg.EncoderDim
		ws.actionEmbeddings[i] = ws.actionEmbeddingData[start : start+cfg.EncoderDim]
	}
	return ws, nil
}

// RankWorkspace owns preallocated input/output vectors and projection scratch.
// Do not share one workspace between concurrent calls.
type RankWorkspace struct {
	engine              *Engine
	candidateCapacity   int
	stateTexts          [1]string
	stateOutputs        [1][]float32
	stateEmbedding      []float32
	actionEmbeddingData []float32
	actionEmbeddings    [][]float32
	scores              []float32
	head                *Workspace
}

// RankInto embeds one state and a batch of candidate actions, computes
// scale*cosine logits, applies a temperature softmax, and writes a stable
// descending ranking into dst. dst must contain at least len(candidates)
// entries; ws must be a workspace created by this Engine.
func (e *Engine) RankInto(ctx context.Context, state string, candidates []string, temperature float32, dst []RankedCandidate, ws *RankWorkspace) error {
	if e == nil || e.Head == nil || e.Embedder == nil {
		return fmt.Errorf("clm: engine is not initialized")
	}
	if ws == nil || ws.engine != e {
		return ErrWorkspace
	}
	n := len(candidates)
	if n == 0 {
		return fmt.Errorf("clm: candidates must not be empty")
	}
	if n > ws.candidateCapacity {
		return fmt.Errorf("clm: workspace holds %d candidates, got %d", ws.candidateCapacity, n)
	}
	if len(dst) < n {
		return fmt.Errorf("clm: ranking buffer has %d entries, want %d", len(dst), n)
	}
	if !finite32(temperature) || temperature <= 0 || temperature > 100 {
		return fmt.Errorf("clm: temperature must be in (0,100], got %g", temperature)
	}
	ws.stateTexts[0] = state
	if err := e.Embedder.Embed(ctx, StateRole, ws.stateTexts[:], ws.stateOutputs[:]); err != nil {
		return fmt.Errorf("clm: embed state: %w", err)
	}
	if err := e.Embedder.Embed(ctx, ActionRole, candidates, ws.actionEmbeddings[:n]); err != nil {
		return fmt.Errorf("clm: embed actions: %w", err)
	}
	if err := e.Head.ScoreInto(ws.stateEmbedding, ws.actionEmbeddings[:n], temperature, ws.scores[:n], ws.head); err != nil {
		return err
	}
	maxLogit := float32(math.Inf(-1))
	for i, score := range ws.scores[:n] {
		if !finite32(score) {
			return fmt.Errorf("clm: non-finite score for candidate %d", i)
		}
		if score > maxLogit {
			maxLogit = score
		}
	}
	var sum float64
	for i, score := range ws.scores[:n] {
		p := math.Exp(float64(score - maxLogit))
		dst[i] = RankedCandidate{Candidate: candidates[i], Probability: float32(p)}
		sum += p
	}
	invSum := 1 / sum
	for i := range dst[:n] {
		dst[i].Probability = float32(float64(dst[i].Probability) * invSum)
	}
	// Stable insertion sort keeps input order when two candidates have the same
	// probability and needs no temporary allocation.
	for i := 1; i < n; i++ {
		item := dst[i]
		j := i
		for j > 0 && dst[j-1].Probability < item.Probability {
			dst[j] = dst[j-1]
			j--
		}
		dst[j] = item
	}
	for i := range dst[:n] {
		dst[i].Rank = i + 1
	}
	return nil
}

// Rank is an allocating convenience wrapper around RankInto.
func (e *Engine) Rank(ctx context.Context, state string, candidates []string, temperature float32) ([]RankedCandidate, error) {
	ws, err := e.NewWorkspace(len(candidates))
	if err != nil {
		return nil, err
	}
	results := make([]RankedCandidate, len(candidates))
	if err := e.RankInto(ctx, state, candidates, temperature, results, ws); err != nil {
		return nil, err
	}
	return results, nil
}
