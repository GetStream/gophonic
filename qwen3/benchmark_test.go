// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

var (
	benchmarkHiddenSink float32
	benchmarkTokenSink  int
)

func BenchmarkOfficialQwenTokenizer(b *testing.B) {
	path := testmodels.Path(b, testmodels.Qwen3)
	tok, err := LoadTokenizer(path)
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
