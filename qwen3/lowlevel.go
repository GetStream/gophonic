// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import "github.com/GetStream/gophonic/internal/qwen3lm"

// The low-level forward pass: load weights, then evaluate pretokenized
// batches with an Evaluator and one Workspace per concurrent call.
type (
	// Weights is an immutable Qwen3 model prepared for inference.
	Weights = qwen3lm.Weights
	// Evaluator computes last-token hidden states over Weights.
	Evaluator = qwen3lm.Evaluator
	// Workspace owns the activations of one concurrent evaluation.
	Workspace = qwen3lm.Workspace
	// PrefixKV stores a token sequence's keys and values for extension.
	PrefixKV = qwen3lm.PrefixKV
	// Tokenizer is the Qwen3 byte-level BPE tokenizer.
	Tokenizer = qwen3lm.Tokenizer
	// TokenizerWorkspace is the reusable scratch of Tokenizer.EncodeInto.
	TokenizerWorkspace = qwen3lm.TokenizerWorkspace
)

// Weight formats accepted by LoadWeights and Options.Weights.
const (
	WeightsF16   = qwen3lm.WeightsF16
	WeightsInt8  = qwen3lm.WeightsInt8
	WeightsGPU   = qwen3lm.WeightsGPU
	WeightsGPUQ4 = qwen3lm.WeightsGPUQ4
)

// Errors of Tokenizer.EncodeInto.
var (
	ErrTokenizerWorkspace = qwen3lm.ErrTokenizerWorkspace
	ErrTokenBufferSmall   = qwen3lm.ErrTokenBufferSmall
)

// LoadWeights reads an official Qwen3 safetensors snapshot directory; see
// the weight format constants.
func LoadWeights(dir, format string) (*Weights, error) { return qwen3lm.LoadWeights(dir, format) }

// NewEvaluator returns an evaluator over m.
func NewEvaluator(m *Weights) (*Evaluator, error) { return qwen3lm.NewEvaluator(m) }

// LoadTokenizer reads tokenizer.json from a snapshot directory or file.
func LoadTokenizer(path string) (*Tokenizer, error) { return qwen3lm.LoadTokenizer(path) }

// QuantizeGPTQ rounds the GPU weights of the snapshot in dir with GPTQ and
// stores them next to it, where LoadWeights finds them.
func QuantizeGPTQ(dir, format string, progress func(layer, layers int)) error {
	return qwen3lm.QuantizeGPTQ(dir, format, progress)
}
