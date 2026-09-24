// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package clmqwen connects CLM's embedding boundary to a local pure-Go Qwen3-8B decoder.
package clmqwen

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"sync"

	"github.com/GetStream/gophonic/clm"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const (
	hiddenSize = 4096
	maxTokens  = 2048
)

// Encoder owns one Qwen3-8B CPU decoder. Int8 mode reuses one FastWorkspace,
// so calls are serialized; use one Encoder per concurrent inference lane.
type Encoder struct {
	mu     sync.Mutex
	model  *decoder.Model
	tokens *tokenizer.Tokenizer
	fast   *FastEvaluator
	ws     *FastWorkspace
	closed bool
}

// Options controls the local Qwen3-8B CPU backend. Quant may be empty for the
// full FP32 path or "int8" for weight-only quantization. Quantized embeddings
// require a separate ranking-accuracy gate against the official checkpoint.
type Options struct{ Quant string }

// Open loads the official Qwen3-8B checkpoint in the full FP32 mode.
func Open(path string) (*Encoder, error) { return OpenWithOptions(path, Options{}) }

// OpenWithOptions loads a Hugging Face safetensors directory or a GGUF file.
func OpenWithOptions(path string, opts Options) (*Encoder, error) {
	if opts.Quant != "" && opts.Quant != "int8" {
		return nil, fmt.Errorf("clmqwen: unsupported quantization %q", opts.Quant)
	}
	var tok *tokenizer.Tokenizer
	var err error
	if strings.EqualFold(filepath.Ext(path), ".gguf") {
		tok, err = tokenizer.LoadGGUF(path)
	} else {
		tok, err = tokenizer.Load(path)
	}
	if err != nil {
		return nil, fmt.Errorf("clmqwen: load tokenizer: %w", err)
	}
	model, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: opts.Quant, ExactPrefill: true})
	if err != nil {
		return nil, fmt.Errorf("clmqwen: load Qwen3: %w", err)
	}
	cfg := model.Config()
	if cfg.ModelType != "qwen3" || cfg.HiddenDim != hiddenSize || cfg.NumLayers != 36 || cfg.NumHeads != 32 || cfg.NumKVHeads != 8 {
		_ = model.Close()
		return nil, fmt.Errorf("clmqwen: expected Qwen3-8B geometry, got type=%q hidden=%d layers=%d heads=%d kv_heads=%d", cfg.ModelType, cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads)
	}
	e := &Encoder{model: model, tokens: tok}
	if opts.Quant == "int8" {
		e.fast, err = NewFastEvaluator(model)
		if err != nil {
			_ = model.Close()
			return nil, fmt.Errorf("clmqwen: initialize fast Qwen3 evaluator: %w", err)
		}
		e.ws = e.fast.NewWorkspace()
	}
	return e, nil
}

// Embed writes raw, post-final-norm, last-token-pooled 4096-dimensional
// hidden states. The CLM head applies L2 normalization exactly once.
// The CLM reference requests vLLM to keep the last 2048 tokens of each text.
// role is deliberately accepted at the whole-batch plug-in boundary; the
// published CLM v0.1 encoder uses the same tokenization for both roles.
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
	for i, input := range texts {
		if err := ctx.Err(); err != nil {
			return err
		}
		ids, err := e.tokens.Encode(input, false)
		if err != nil {
			return fmt.Errorf("clmqwen: tokenize input %d: %w", i, err)
		}
		if err := e.embedIDsLocked(ctx, i, ids, dst[i]); err != nil {
			return err
		}
	}
	return nil
}

// EmbedTokensInto embeds caller-tokenized inputs into caller-owned vectors.
// With the int8 backend, this is allocation-free after each workspace has
// warmed to the longest input. Text Embed remains the convenience API and pays
// the tokenizer's allocations. FP32 keeps the existing GoInfer reference path.
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
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("clmqwen: encoder closed")
	}
	width := hiddenSize
	if e.fast != nil {
		width = e.fast.hidden
	} else if e.model != nil {
		width = e.model.Config().HiddenDim
	}
	for _, row := range dst {
		if len(row) != width {
			return fmt.Errorf("clmqwen: destination width %d, want %d", len(row), width)
		}
	}
	for i, ids := range tokenIDs {
		if err := e.embedIDsLocked(ctx, i, ids, dst[i]); err != nil {
			return err
		}
	}
	return nil
}

func (e *Encoder) embedIDsLocked(ctx context.Context, index int, ids []int, dst []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ids) == 0 {
		return fmt.Errorf("clmqwen: input %d produced no tokens", index)
	}
	if len(ids) > maxTokens {
		ids = ids[len(ids)-maxTokens:]
	}
	if e.fast != nil {
		if err := e.fast.HiddenLastInto(ids, dst, e.ws); err != nil {
			return fmt.Errorf("clmqwen: infer input %d: %w", index, err)
		}
		return nil
	}
	hidden, err := e.model.HiddenLast(ids)
	if err != nil {
		return fmt.Errorf("clmqwen: infer input %d: %w", index, err)
	}
	if err := copyHidden(dst, hidden); err != nil {
		return fmt.Errorf("clmqwen: copy hidden input %d: %w", index, err)
	}
	return nil
}

func copyHidden(dst, src []float32) error {
	if len(src) != hiddenSize {
		return fmt.Errorf("hidden width %d, want %d", len(src), hiddenSize)
	}
	for _, value := range src {
		if !finite32(value) {
			return errors.New("non-finite hidden state")
		}
	}
	copy(dst, src)
	return nil
}

func finite32(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

// Close releases the decoder's weights and backend resources. It waits for an
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
	err := e.model.Close()
	e.fast = nil
	e.ws = nil
	e.model = nil
	return err
}
