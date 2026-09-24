// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"context"
	"os"
	"testing"

	"github.com/GetStream/gophonic/clm"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// BenchmarkOfficialQwenEmbedding measures fresh one-token text inference on the
// published Qwen3-8B weights. Loading is excluded, but tokenization, decoder
// execution, and the caller-facing copy are included. Set GOPHONIC_QWEN3_MODEL
// to the local safetensors directory to run it.
func BenchmarkOfficialQwenEmbedding(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_MODEL to the official Qwen3-8B directory")
	}
	encoder, err := OpenWithOptions(path, Options{Quant: "int8"})
	if err != nil {
		b.Fatal(err)
	}
	defer encoder.Close()
	text := []string{"hello"}
	out := [][]float32{make([]float32, hiddenSize)}
	if err := encoder.Embed(context.Background(), clm.StateRole, text, out); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := encoder.Embed(context.Background(), clm.StateRole, text, out); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	benchmarkHiddenSink = out[0][0]
}

var benchmarkHiddenSink float32

func BenchmarkOfficialQwenTokenizer(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_TOKENIZER")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_TOKENIZER to the official Qwen3-8B directory")
	}
	tok, err := tokenizer.Load(path)
	if err != nil {
		b.Fatal(err)
	}
	var ids []int
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ids, err = tok.Encode("hello", false)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	benchmarkTokenSink = len(ids)
}

var benchmarkTokenSink int
