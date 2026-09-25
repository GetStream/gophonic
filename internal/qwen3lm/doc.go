// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package qwen3lm is the Qwen3 dense transformer that gophonic's Qwen3
// models share: the safetensors loader and weight formats, the byte-level
// BPE tokenizer, and the batched forward pass with stored key/value
// prefixes, on SME matrix tiles, portable CPU kernels, or the Apple GPU.
// Package qwen3 builds text tasks on it and package qwen3asr its speech
// decoder.
package qwen3lm
