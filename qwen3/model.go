// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package qwen3 runs Qwen3 dense transformers (Qwen3-8B, 4B, 1.7B, 0.6B) in
// pure Go, on the CPU or the Apple GPU.
//
// It loads an official Hugging Face safetensors snapshot, tokenizes text,
// and computes the final-normalized hidden state of each input's last token.
// On Apple M4 it uses SME matrix tiles; other CPUs use portable kernels.
// Model is the high-level entry point: it batches inputs, caches finished
// vectors, reuses stored key/value prefixes, and answers multiple-choice
// questions with Choose. Evaluator and Workspace expose the forward pass
// directly for pretokenized batches.
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

// Model owns one Qwen3 model and one inference workspace, so calls are
// serialized; use one Model per concurrent inference lane.
type Model struct {
	mu          sync.Mutex
	model       *Weights
	tokens      *Tokenizer
	tokenWS     TokenizerWorkspace
	tokenBufs   [][]int // per-text token IDs for a batched Embed call
	batchIDs    [][]int
	missIDs     [][]int // inputs not served by the cache
	missDst     [][]float32
	shortIDs    [][]int // inputs batched together (below prefixMinTokens)
	shortDst    [][]float32
	letters     *letterHead // answer-letter head rows for Question, or nil
	answer      string      // what opens the assistant's answer (answerThinking or answerPlain)
	cache       *embeddingCache
	prefix      *PrefixKV // last long input's keys and values, or nil
	reused      uint64    // tokens served from prefix
	computed    uint64    // tokens evaluated for long inputs
	eval        *Evaluator
	ws          *Workspace
	closed      bool
	ownsWeights bool // loaded by Open, so Close frees their GPU memory
}

// Options controls the local Qwen3 backend. The zero value picks the
// fastest backend with llama.cpp Q8_0 fidelity: on an Apple GPU, WeightsGPU
// (int8 rows, FP32 activations) for Qwen3-8B and WeightsGPUQ8 (int8 blocks of
// 32) for other sizes; elsewhere WeightsF16 on the CPU (every BF16 checkpoint
// weight exactly). Weights may name a backend explicitly: WeightsF16,
// WeightsInt8 (CPU, per-row int8), WeightsGPU, WeightsGPUQ8, or WeightsGPUQ4
// (4.5-bit GPU weights, lowest latency, lower fidelity).
// Threads bounds the CPU worker goroutines, including the caller; zero
// selects min(performance cores, GOMAXPROCS). CacheEntries sizes an exact
// cache of finished embeddings keyed by token IDs (16 KiB per entry for
// Qwen3-8B): zero
// selects 4096 entries, and a negative value disables it. A hit returns
// exactly the vector a fresh evaluation would produce.
//
// PrefixCacheTokens sizes a store of the last long input's per-layer keys and
// values (288 KiB per token for Qwen3-8B): when a later input of at least 64 tokens shares
// a token prefix with it (a growing conversation state), only the new tokens
// are evaluated. Zero selects 2048 tokens; a negative value disables it.
type Options struct {
	Weights           string
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

// Open loads an official Qwen3 safetensors snapshot directory, such as
// Qwen/Qwen3-8B or Qwen/Qwen3-1.7B. The zero Options value selects the
// fastest faithful backend and default caches.
func Open(path string, opts Options) (*Model, error) {
	if opts.Threads < 0 {
		return nil, fmt.Errorf("qwen3: invalid thread count %d", opts.Threads)
	}
	tokens, err := LoadTokenizer(path)
	if err != nil {
		return nil, fmt.Errorf("qwen3: load tokenizer: %w", err)
	}
	model, err := LoadWeights(path, opts.Weights)
	if err != nil {
		return nil, err
	}
	entries := opts.CacheEntries
	if entries == 0 {
		entries = defaultCacheEntries
	}
	prefix := opts.PrefixCacheTokens
	if prefix == 0 {
		prefix = maxTokens
	}
	letters, err := loadLetterHead(path, tokens, model.Config().Hidden, model.Config().Vocab)
	if err != nil {
		return nil, err
	}
	e, err := newModel(model, tokens, opts.threads(), entries, min(prefix, maxTokens, model.Config().MaxPositions))
	if err != nil {
		return nil, err
	}
	e.letters = letters
	if !thinks(path) {
		e.answer = answerPlain
	}
	e.ownsWeights = true
	return e, nil
}

func newModel(model *Weights, tokens *Tokenizer, threads, cacheEntries, prefixTokens int) (*Model, error) {
	eval, err := NewEvaluator(model)
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
	e := &Model{model: model, tokens: tokens, eval: eval, ws: ws, cache: newEmbeddingCache(cacheEntries, model.Config().Hidden), answer: answerThinking}
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
func (e *Model) Width() int { return e.model.Config().Hidden }

// PrefixStats reports, for inputs of at least 64 tokens, how many tokens were
// served from the prefix store and how many were evaluated.
func (e *Model) PrefixStats() (reused, computed uint64) {
	if e == nil {
		return 0, 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.reused, e.computed
}

// CacheStats reports embedding-cache hits and lookups since Open.
func (e *Model) CacheStats() (hits, lookups uint64) {
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
func (e *Model) Embed(ctx context.Context, texts []string, dst [][]float32) error {
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
func (e *Model) EmbedTokensInto(ctx context.Context, tokenIDs [][]int, dst [][]float32) error {
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
func (e *Model) embedBatchLocked(ctx context.Context, ids [][]int, dst [][]float32) error {
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

func (e *Model) inferLocked(ctx context.Context, ids [][]int, dst [][]float32) error {
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

// Close stops the encoder's workers and releases its model. It waits for an
// in-flight Embed or EmbedTokensInto call. A closed Model cannot be reused.
func (e *Model) Close() error {
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
	if e.ownsWeights {
		e.model.Release()
	}
	e.ws, e.eval, e.model, e.tokens = nil, nil, nil, nil
	e.tokenBufs, e.batchIDs, e.missIDs, e.missDst, e.cache, e.prefix = nil, nil, nil, nil, nil, nil
	e.shortIDs, e.shortDst = nil, nil
	return err
}
