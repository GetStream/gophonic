// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func BenchmarkFastOfficialHiddenLastInto(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_FAST_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_FAST_MODEL to the official Qwen3-8B directory")
	}
	model, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "int8", ExactPrefill: true})
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	f, err := NewFastEvaluator(model)
	if err != nil {
		b.Fatal(err)
	}
	tok, err := tokenizer.Load(path)
	if err != nil {
		b.Fatal(err)
	}
	ids, err := tok.Encode("hello", false)
	if err != nil {
		b.Fatal(err)
	}
	ws, out := f.NewWorkspace(), make([]float32, f.hidden)
	if err := f.HiddenLastInto(ids, out, ws); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := f.HiddenLastInto(ids, out, ws); err != nil {
			b.Fatal(err)
		}
	}
	benchmarkHiddenSink = out[0]
}

func BenchmarkGoinferOfficialHiddenLast(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_FAST_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_FAST_MODEL to the official Qwen3-8B directory")
	}
	model, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "int8", ExactPrefill: true})
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	tok, err := tokenizer.Load(path)
	if err != nil {
		b.Fatal(err)
	}
	ids, err := tok.Encode("hello", false)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		out, err := model.HiddenLast(ids)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkHiddenSink = out[0]
	}
}
