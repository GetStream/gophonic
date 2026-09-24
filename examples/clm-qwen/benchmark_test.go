// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"os"
	"testing"
)

var (
	benchmarkHiddenSink float32
	benchmarkTokenSink  int
)

func BenchmarkOfficialQwenTokenizer(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_TOKENIZER")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_TOKENIZER to the official Qwen3-8B directory")
	}
	tok, err := LoadQwenTokenizer(path)
	if err != nil {
		b.Fatal(err)
	}
	var ws TokenizerWorkspace
	ids := make([]int, 0, 64)
	if ids, err = tok.EncodeInto("hello", ids, &ws); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		ids, err = tok.EncodeInto("hello", ids[:0], &ws)
		if err != nil {
			b.Fatal(err)
		}
	}
	benchmarkTokenSink = len(ids)
}
