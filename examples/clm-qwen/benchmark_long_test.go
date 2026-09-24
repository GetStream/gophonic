// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"context"
	"os"
	"testing"

	"github.com/GetStream/gophonic/clm"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func BenchmarkFastOfficialTwelveTokens(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_FAST_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_FAST_MODEL")
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
	ids, err := tok.Encode("The Moon causes tides by pulling on Earth's oceans.", false)
	if err != nil {
		b.Fatal(err)
	}
	if len(ids) != 12 {
		b.Fatalf("tokenizer returned %d tokens, want 12", len(ids))
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
	b.StopTimer()
	benchmarkHiddenSink = out[0]
}

// BenchmarkOfficialQwenTwelveTokenEmbedding includes public text tokenization
// and layer-batched inference. This is the end-to-end 500 ms target boundary.
func BenchmarkOfficialQwenTwelveTokenEmbedding(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_FAST_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_FAST_MODEL")
	}
	encoder, err := OpenWithOptions(path, Options{Quant: "int8"})
	if err != nil {
		b.Fatal(err)
	}
	defer encoder.Close()
	text := []string{"The Moon causes tides by pulling on Earth's oceans."}
	out := [][]float32{make([]float32, hiddenSize)}
	ctx := context.Background()
	if err := encoder.Embed(ctx, clm.StateRole, text, out); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := encoder.Embed(ctx, clm.StateRole, text, out); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	benchmarkHiddenSink = out[0][0]
}
