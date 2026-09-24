// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package clmqwen connects CLM's embedding boundary to a local pure-Go Qwen3-8B decoder.
package clmqwen

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/GetStream/gophonic/clm"
)

const (
	hiddenSize = 4096
	maxTokens  = 2048
	// batchTokens bounds the tokens packed into one forward pass. Texts in
	// one Embed call share 16-row matrix tiles up to this budget; a single
	// longer text runs alone.
	batchTokens = 512
	// defaultMaxThreads caps the default worker count. On an M4 Max, the two
	// performance clusters' SME units saturate near eight streaming threads;
	// more threads only add contention to every layer's barriers.
	defaultMaxThreads = 8
	// defaultCacheEntries sizes the exact embedding cache (16 KiB each).
	defaultCacheEntries = 4096
)

// Encoder owns one Qwen3-8B model and one inference workspace, so calls are
// serialized; use one Encoder per concurrent inference lane.
type Encoder struct {
	mu        sync.Mutex
	model     *Model
	tokens    *QwenTokenizer
	tokenWS   TokenizerWorkspace
	tokenBufs [][]int // per-text token IDs for a batched Embed call
	batchIDs  [][]int
	missIDs   [][]int // inputs not served by the cache
	missDst   [][]float32
	cache     *embeddingCache
	eval      *Evaluator
	ws        *Workspace
	closed    bool
}

// Options controls the local Qwen3-8B CPU backend. Weights selects WeightsF16
// (the default: every BF16 checkpoint weight exactly) or WeightsInt8 (per-row
// int8, half the memory, lower fidelity). Threads bounds the worker
// goroutines, including the caller; zero selects min(performance cores,
// GOMAXPROCS, 8). CacheEntries sizes an exact cache of finished embeddings
// keyed by token IDs (16 KiB per entry): zero selects 4096 entries, and a
// negative value disables it. A hit returns exactly the vector a fresh
// evaluation would produce.
type Options struct {
	Weights      string
	Threads      int
	CacheEntries int
}

func (o Options) threads() int {
	if o.Threads > 0 {
		return min(o.Threads, 64)
	}
	n := performanceCores()
	if n <= 0 {
		n = runtime.NumCPU()
	}
	return max(1, min(n, runtime.GOMAXPROCS(0), defaultMaxThreads))
}

// Open loads an official Qwen3-8B safetensors snapshot with exact weights.
func Open(path string) (*Encoder, error) { return OpenWithOptions(path, Options{}) }

// OpenWithOptions loads an official Qwen3-8B safetensors snapshot directory.
func OpenWithOptions(path string, opts Options) (*Encoder, error) {
	if opts.Threads < 0 {
		return nil, fmt.Errorf("clmqwen: invalid thread count %d", opts.Threads)
	}
	tokens, err := LoadQwenTokenizer(path)
	if err != nil {
		return nil, fmt.Errorf("clmqwen: load tokenizer: %w", err)
	}
	model, err := LoadModel(path, opts.Weights)
	if err != nil {
		return nil, err
	}
	c := model.cfg
	if c.hidden != hiddenSize || c.layers != 36 || c.heads != 32 || c.kvHeads != 8 {
		return nil, fmt.Errorf("clmqwen: expected Qwen3-8B geometry, got hidden=%d layers=%d heads=%d kv_heads=%d", c.hidden, c.layers, c.heads, c.kvHeads)
	}
	entries := opts.CacheEntries
	if entries == 0 {
		entries = defaultCacheEntries
	}
	return newEncoder(model, tokens, opts.threads(), entries)
}

func newEncoder(model *Model, tokens *QwenTokenizer, threads, cacheEntries int) (*Encoder, error) {
	eval, err := NewEvaluator(model)
	if err != nil {
		return nil, err
	}
	ws, err := eval.NewWorkspace(threads)
	if err != nil {
		return nil, err
	}
	return &Encoder{model: model, tokens: tokens, eval: eval, ws: ws, cache: newEmbeddingCache(cacheEntries, model.cfg.hidden)}, nil
}

// CacheStats reports embedding-cache hits and lookups since Open.
func (e *Encoder) CacheStats() (hits, lookups uint64) {
	if e == nil {
		return 0, 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.cache == nil {
		return 0, 0
	}
	return e.cache.hits, e.cache.all
}

// Embed writes raw, post-final-norm, last-token-pooled 4096-dimensional
// hidden states. The CLM head applies L2 normalization exactly once.
// The CLM reference requests vLLM to keep the last 2048 tokens of each text.
// role is deliberately accepted at the whole-batch plug-in boundary; the
// published CLM v0.1 encoder uses the same tokenization for both roles.
// Texts of one call are packed into shared forward passes. After the
// workspaces have warmed to the call's shape, Embed allocates nothing.
func (e *Encoder) Embed(ctx context.Context, role clm.Role, texts []string, dst [][]float32) error {
	if e == nil {
		return errors.New("clmqwen: nil encoder")
	}
	if role != clm.StateRole && role != clm.ActionRole {
		return errors.New("clmqwen: invalid role")
	}
	if len(dst) != len(texts) {
		return fmt.Errorf("clmqwen: %d destination vectors for %d texts", len(dst), len(texts))
	}
	for _, row := range dst {
		if len(row) != hiddenSize {
			return fmt.Errorf("clmqwen: destination width %d, want %d", len(row), hiddenSize)
		}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("clmqwen: encoder closed")
	}
	if len(e.tokenBufs) < len(texts) {
		e.tokenBufs = append(e.tokenBufs, make([][]int, len(texts)-len(e.tokenBufs))...)
	}
	// batchIDs holds views that truncation may reslice; tokenBufs keeps
	// each buffer's full capacity for the next call.
	e.batchIDs = e.batchIDs[:0]
	for i, input := range texts {
		buf := e.tokenBufs[i]
		// Byte-level BPE emits at most one token per input byte for this
		// checkpoint, so a buffer grows only for a longer input.
		if len(input) > cap(buf) {
			buf = make([]int, 0, len(input))
		}
		tokens, err := e.tokens.EncodeInto(input, buf[:0], &e.tokenWS)
		if err != nil {
			return fmt.Errorf("clmqwen: tokenize input %d: %w", i, err)
		}
		e.tokenBufs[i] = tokens
		e.batchIDs = append(e.batchIDs, tokens)
	}
	err := e.embedBatchLocked(ctx, e.batchIDs, dst)
	clear(e.batchIDs)
	return err
}

// EmbedTokensInto embeds caller-tokenized inputs into caller-owned vectors.
// It is allocation-free after the workspace has warmed to the call's shape.
func (e *Encoder) EmbedTokensInto(ctx context.Context, role clm.Role, tokenIDs [][]int, dst [][]float32) error {
	if e == nil {
		return errors.New("clmqwen: nil encoder")
	}
	if role != clm.StateRole && role != clm.ActionRole {
		return errors.New("clmqwen: invalid role")
	}
	if len(dst) != len(tokenIDs) {
		return fmt.Errorf("clmqwen: %d destination vectors for %d token sequences", len(dst), len(tokenIDs))
	}
	for _, row := range dst {
		if len(row) != hiddenSize {
			return fmt.Errorf("clmqwen: destination width %d, want %d", len(row), hiddenSize)
		}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("clmqwen: encoder closed")
	}
	// Truncation reslices; keep the caller's slice headers untouched.
	e.batchIDs = append(e.batchIDs[:0], tokenIDs...)
	err := e.embedBatchLocked(ctx, e.batchIDs, dst)
	clear(e.batchIDs)
	return err
}

// embedBatchLocked serves cached inputs, then packs the rest into forward
// passes of at most batchTokens tokens (or one longer input), keeping the last
// maxTokens of each.
func (e *Encoder) embedBatchLocked(ctx context.Context, ids [][]int, dst [][]float32) error {
	for i, seq := range ids {
		if len(seq) == 0 {
			return fmt.Errorf("clmqwen: input %d produced no tokens", i)
		}
		if len(seq) > maxTokens {
			ids[i] = seq[len(seq)-maxTokens:]
		}
	}
	if e.cache == nil {
		return e.inferLocked(ctx, ids, dst)
	}
	e.missIDs, e.missDst = e.missIDs[:0], e.missDst[:0]
	for i, seq := range ids {
		if !e.cache.get(seq, dst[i]) {
			e.missIDs = append(e.missIDs, seq)
			e.missDst = append(e.missDst, dst[i])
		}
	}
	err := e.inferLocked(ctx, e.missIDs, e.missDst)
	if err == nil {
		for i, seq := range e.missIDs {
			e.cache.put(seq, e.missDst[i])
		}
	}
	clear(e.missIDs)
	clear(e.missDst)
	return err
}

func (e *Encoder) inferLocked(ctx context.Context, ids [][]int, dst [][]float32) error {
	for start := 0; start < len(ids); {
		if err := ctx.Err(); err != nil {
			return err
		}
		end, tokens := start, 0
		for end < len(ids) && (end == start || tokens+len(ids[end]) <= batchTokens) {
			tokens += len(ids[end])
			end++
		}
		if err := e.eval.HiddenLastBatchInto(ids[start:end], dst[start:end], e.ws); err != nil {
			return fmt.Errorf("clmqwen: infer inputs %d-%d: %w", start, end-1, err)
		}
		start = end
	}
	return nil
}

func finite32(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

// Close stops the encoder's workers and releases its model. It waits for an
// in-flight Embed or EmbedTokensInto call. A closed Encoder cannot be reused.
func (e *Encoder) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	err := e.ws.Close()
	e.ws, e.eval, e.model, e.tokens = nil, nil, nil, nil
	e.tokenBufs, e.batchIDs, e.missIDs, e.missDst, e.cache = nil, nil, nil, nil, nil
	return err
}
