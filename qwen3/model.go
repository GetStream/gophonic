// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package qwen3 runs the Qwen3 family in pure Go, on the CPU or the Apple
// GPU: the dense models (Qwen3-8B, 4B, 1.7B, 0.6B), the mixtures of experts
// (Qwen3-30B-A3B), and the Qwen3.6 hybrid (Qwen3.6-35B-A3B).
//
// Open loads an official Hugging Face safetensors snapshot as one Model
// that converses (chat.Generator, with tools in Qwen3's own format),
// answers multiple-choice questions about text with a single prefill
// (Question, Classifier, NewContext), and embeds text (Embed), each prepared
// at its first use. On Apple M4 the CPU path uses SME matrix tiles; other
// CPUs use portable kernels. Tokenizer is public for callers that tokenize
// themselves.
package qwen3

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/thesyncim/vibejson"
)

const (
	maxTokens = 2048
	// batchTokens bounds the tokens packed into one forward pass. Texts in
	// one Embed call share 16-row matrix tiles up to this budget; a single
	// longer text runs alone.
	batchTokens = 512
	// defaultMaxThreads caps the default worker count. At most eight workers
	// run SME at once (more only add contention); the rest take NEON strips
	// of multi-tile int8 projections and share the elementwise stages.
	defaultMaxThreads = 16
	// defaultCacheEntries sizes the exact embedding cache (16 KiB each).
	defaultCacheEntries = 4096
	// prefixMinTokens is the text length from which inputs are evaluated
	// through the prefix store instead of the shared short-text batch.
	prefixMinTokens = 64
)

// encoder is a Model's questions and embeddings: one workspace, whose
// calls are serialized, a prefix store for long inputs, and an embedding
// cache.
type encoder struct {
	mu        sync.Mutex
	model     *qwen3lm.Weights
	tokens    *Tokenizer
	tokenWS   TokenizerWorkspace
	tokenBufs [][]int // per-text token IDs for a batched Embed call
	batchIDs  [][]int
	missIDs   [][]int // inputs not served by the cache
	missDst   [][]float32
	shortIDs  [][]int // inputs batched together (below prefixMinTokens)
	shortDst  [][]float32
	letters   *letterHead // answer-letter head rows for Question, or nil
	answer    string      // what opens the assistant's answer (answerThinking or answerPlain)
	cache     *embeddingCache
	prefix    *qwen3lm.PrefixKV // last long input's keys and values, or nil
	reused    uint64            // tokens served from prefix
	computed  uint64            // tokens evaluated for long inputs
	eval      *qwen3lm.Evaluator
	ws        *qwen3lm.Workspace
	closed    bool
}

// Options controls the local Qwen3 backend.
//
// Format names the weight format, as gophonic.Options does: "f16" (every
// BF16 checkpoint weight exactly, on the CPU), "int8" (per-row int8 on the
// CPU), "gpu" (int8 rows, FP32 activations, on the Apple GPU), "gpu-q8"
// (int8 blocks of 32), or "gpu-q4" (4.5-bit blocks: lowest latency, lower
// fidelity). Empty picks the fastest of llama.cpp Q8_0 fidelity: on an
// Apple GPU, "gpu" for Qwen3-8B and "gpu-q8" for other sizes; elsewhere
// "f16".
//
// Threads bounds the CPU worker goroutines, including the caller; zero
// selects min(performance cores, GOMAXPROCS). CacheEntries sizes an exact
// cache of finished embeddings keyed by token IDs (16 KiB per entry for
// Qwen3-8B): zero selects 4096 entries, and a negative value disables it.
// A hit returns exactly the vector a fresh evaluation would produce.
//
// PrefixCacheTokens sizes a store of the last long input's per-layer keys and
// values (288 KiB per token for Qwen3-8B): when a later input of at least 64 tokens shares
// a token prefix with it (a growing conversation state), only the new tokens
// are evaluated. Zero selects 2048 tokens; a negative value disables it.
type Options struct {
	Format            string
	Threads           int
	CacheEntries      int
	PrefixCacheTokens int
}

func (o Options) threads() int {
	if o.Threads > 0 {
		return min(o.Threads, 64)
	}
	n := qwen3lm.PerformanceCores()
	if n <= 0 {
		n = runtime.NumCPU()
	}
	return max(1, min(n, runtime.GOMAXPROCS(0), defaultMaxThreads))
}

// IsModelDir reports whether dir holds a Qwen3 checkpoint: a config.json
// whose model_type is qwen3, qwen3_moe for a mixture of experts, or
// qwen3_5_moe for the Qwen3.5 family (Qwen3.6-35B-A3B).
func IsModelDir(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	if vibejson.Unmarshal(raw, &c) != nil {
		return false
	}
	switch c.ModelType {
	case "qwen3", "qwen3_moe", "qwen3_5_moe":
		return true
	}
	return false
}

// newEncoder prepares a Model's questions and embeddings over its weights.
func newEncoder(weights *qwen3lm.Weights, tokens *Tokenizer, path string, opts Options) (*encoder, error) {
	cfg := weights.Config()
	entries := opts.CacheEntries
	if entries == 0 {
		entries = defaultCacheEntries
	}
	prefix := opts.PrefixCacheTokens
	if prefix == 0 {
		prefix = maxTokens
	}
	letters, err := loadLetterHead(path, tokens, cfg.Hidden, cfg.Vocab)
	if err != nil {
		return nil, err
	}
	e, err := newModel(weights, tokens, opts.threads(), entries, min(prefix, maxTokens, cfg.MaxPositions))
	if err != nil {
		return nil, err
	}
	e.letters = letters
	if !thinks(path) {
		e.answer = answerPlain
	}
	return e, nil
}

func newModel(model *qwen3lm.Weights, tokens *Tokenizer, threads, cacheEntries, prefixTokens int) (*encoder, error) {
	eval, err := qwen3lm.NewEvaluator(model)
	if err != nil {
		return nil, err
	}
	ws, err := eval.NewWorkspace(threads)
	if err != nil {
		return nil, err
	}
	// Allocate the whole working set now: short-text batches up to
	// batchTokens rows and positions up to maxTokens. Hot-path calls within
	// those bounds never allocate; a single input longer than batchTokens
	// grows the row storage once.
	if err := ws.Reserve(min(batchTokens, model.Config().MaxPositions), min(maxTokens, model.Config().MaxPositions)); err != nil {
		_ = ws.Close()
		return nil, err
	}
	e := &encoder{model: model, tokens: tokens, eval: eval, ws: ws, cache: newEmbeddingCache(cacheEntries, model.Config().Hidden), answer: answerThinking}
	if prefixTokens > 0 {
		if e.prefix, err = eval.NewPrefixKV(prefixTokens); err != nil {
			_ = ws.Close()
			return nil, err
		}
	}
	return e, nil
}

// Width reports the length of an embedding: the model's hidden size, 4096
// for Qwen3-8B.
func (e *encoder) Width() int { return e.model.Config().Hidden }

// PrefixStats reports, for inputs of at least 64 tokens, how many tokens were
// served from the prefix store and how many were evaluated.
func (e *encoder) PrefixStats() (reused, computed uint64) {
	if e == nil {
		return 0, 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.reused, e.computed
}

// CacheStats reports embedding-cache hits and lookups since Open.
func (e *encoder) CacheStats() (hits, lookups uint64) {
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

// Embed writes each text's raw, post-final-norm, last-token hidden state
// (the model's width: 4096 values for Qwen3-8B) to the matching dst row, keeping the last 2048 tokens of
// longer texts as the CLM reference does. No special tokens are added. Texts
// of one call are packed into shared forward passes. After the workspaces
// have warmed to the call's shape, Embed allocates nothing.
//
// Embed has the shape of a clm.Embedder without its role argument; the
// published CLM heads use the same encoding for states and actions:
//
//	clm.EmbedFunc(func(ctx context.Context, _ clm.Role, texts []string, dst [][]float32) error {
//		return enc.Embed(ctx, texts, dst)
//	})
func (e *encoder) Embed(ctx context.Context, texts []string, dst [][]float32) error {
	if e == nil {
		return errors.New("qwen3: nil encoder")
	}
	if len(dst) != len(texts) {
		return fmt.Errorf("qwen3: %d destination vectors for %d texts", len(dst), len(texts))
	}
	for _, row := range dst {
		if len(row) != e.model.Config().Hidden {
			return fmt.Errorf("qwen3: destination width %d, want %d", len(row), e.model.Config().Hidden)
		}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("qwen3: encoder closed")
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
			return fmt.Errorf("qwen3: tokenize input %d: %w", i, err)
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
func (e *encoder) EmbedTokensInto(ctx context.Context, tokenIDs [][]int, dst [][]float32) error {
	if e == nil {
		return errors.New("qwen3: nil encoder")
	}
	if len(dst) != len(tokenIDs) {
		return fmt.Errorf("qwen3: %d destination vectors for %d token sequences", len(dst), len(tokenIDs))
	}
	for _, row := range dst {
		if len(row) != e.model.Config().Hidden {
			return fmt.Errorf("qwen3: destination width %d, want %d", len(row), e.model.Config().Hidden)
		}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("qwen3: encoder closed")
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
func (e *encoder) embedBatchLocked(ctx context.Context, ids [][]int, dst [][]float32) error {
	for i, seq := range ids {
		if len(seq) == 0 {
			return fmt.Errorf("qwen3: input %d produced no tokens", i)
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

func (e *encoder) inferLocked(ctx context.Context, ids [][]int, dst [][]float32) error {
	if e.prefix != nil {
		// Long inputs run alone through the prefix store; the rest keep
		// sharing batched forward passes.
		e.shortIDs, e.shortDst = e.shortIDs[:0], e.shortDst[:0]
		defer func() {
			clear(e.shortIDs)
			clear(e.shortDst)
		}()
		for i, seq := range ids {
			if len(seq) < prefixMinTokens || len(seq) > e.prefix.Capacity() {
				e.shortIDs = append(e.shortIDs, seq)
				e.shortDst = append(e.shortDst, dst[i])
				continue
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			keep := min(e.prefix.CommonPrefix(seq), len(seq)-1)
			if err := e.eval.HiddenLastExtendInto(e.prefix, keep, seq[keep:], dst[i], e.ws); err != nil {
				return fmt.Errorf("qwen3: infer input %d: %w", i, err)
			}
			e.reused += uint64(keep)
			e.computed += uint64(len(seq) - keep)
		}
		ids, dst = e.shortIDs, e.shortDst
	}
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
			return fmt.Errorf("qwen3: infer inputs %d-%d: %w", start, end-1, err)
		}
		start = end
	}
	return nil
}

// Close stops the encoder's workers. It waits for an in-flight Embed or
// EmbedTokensInto call.
func (e *encoder) Close() error {
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
	e.tokenBufs, e.batchIDs, e.missIDs, e.missDst, e.cache, e.prefix = nil, nil, nil, nil, nil, nil
	e.shortIDs, e.shortDst = nil, nil
	return err
}
