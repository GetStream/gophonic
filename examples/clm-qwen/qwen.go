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

// Encoder owns one Qwen3-8B CPU decoder. Calls are serialized because the
// decoder's forward pass owns mutable scratch and a fresh KV cache.
// A separate Encoder is needed for each concurrent inference lane.
type Encoder struct {
	mu     sync.Mutex
	model  *decoder.Model
	tokens *tokenizer.Tokenizer
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
	return &Encoder{model: model, tokens: tok}, nil
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
		if len(ids) == 0 {
			return fmt.Errorf("clmqwen: input %d produced no tokens", i)
		}
		if len(ids) > maxTokens {
			ids = ids[len(ids)-maxTokens:]
		}
		hidden, err := e.model.HiddenLast(ids)
		if err != nil {
			return fmt.Errorf("clmqwen: infer input %d: %w", i, err)
		}
		if err := copyHidden(dst[i], hidden); err != nil {
			return fmt.Errorf("clmqwen: copy hidden input %d: %w", i, err)
		}
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
// in-flight Embed call. A closed Encoder cannot be reused.
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
	return e.model.Close()
}
