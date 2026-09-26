// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import "github.com/GetStream/gophonic/internal/qwen3lm"

// Tokenizers: applications tokenize for EmbedTokensInto, and count tokens.
type (
	// Tokenizer is the Qwen3 byte-level BPE tokenizer.
	Tokenizer = qwen3lm.Tokenizer
	// TokenizerWorkspace is the reusable scratch of Tokenizer.EncodeInto.
	TokenizerWorkspace = qwen3lm.TokenizerWorkspace
)

// Errors of Tokenizer.EncodeInto.
var (
	ErrTokenizerWorkspace = qwen3lm.ErrTokenizerWorkspace
	ErrTokenBufferSmall   = qwen3lm.ErrTokenBufferSmall
)

// LoadTokenizer reads tokenizer.json from a snapshot directory or file.
func LoadTokenizer(path string) (*Tokenizer, error) { return qwen3lm.LoadTokenizer(path) }
