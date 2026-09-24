// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestPrefillMatchesTokenMajorAndAllocatesZero(t *testing.T) {
	fast, err := tinyFastEvaluator()
	if err != nil {
		t.Fatal(err)
	}
	prefill := newPrefillEvaluator(fast)
	fastWS, prefillWS := fast.NewWorkspace(), prefill.NewWorkspace()
	for _, ids := range [][]int{{1}, {1, 2, 3, 0}, {1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 2, 3}} {
		got, want := make([]float32, fast.hidden), make([]float32, fast.hidden)
		if err := fast.HiddenLastInto(ids, want, fastWS); err != nil {
			t.Fatal(err)
		}
		if err := prefill.HiddenLastInto(ids, got, prefillWS); err != nil {
			t.Fatal(err)
		}
		cos, maxAbs := vectorParity(got, want)
		if cos < 0.99999 || maxAbs > 1e-5 {
			t.Fatalf("tokens=%d: prefill diverged from token-major path: cosine=%.9f max_abs=%g", len(ids), cos, maxAbs)
		}
	}
	ids := []int{1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 2, 3}
	out := make([]float32, fast.hidden)
	if got := testing.AllocsPerRun(100, func() {
		if err := prefill.HiddenLastInto(ids, out, prefillWS); err != nil {
			panic(err)
		}
	}); got != 0 {
		t.Fatalf("warmed 12-token prefill allocated %.2f times/call, want zero", got)
	}
}

func TestPrefillOfficialQwenParity(t *testing.T) {
	path := os.Getenv("GOPHONIC_QWEN3_FAST_MODEL")
	if path == "" {
		t.Skip("set GOPHONIC_QWEN3_FAST_MODEL to the official Qwen3-8B directory")
	}
	model, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "int8", ExactPrefill: true})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	fast, err := NewFastEvaluator(model)
	if err != nil {
		t.Fatal(err)
	}
	prefill := newPrefillEvaluator(fast)
	fastWS, prefillWS := fast.NewWorkspace(), prefill.NewWorkspace()
	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	ids12, err := tok.Encode("The Moon causes tides by pulling on Earth's oceans.", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids12) != 12 {
		t.Fatalf("official phrase encoded to %d tokens, want 12", len(ids12))
	}
	for _, ids := range [][]int{ids12[:1], ids12[:4], ids12} {
		want, err := model.HiddenLast(ids)
		if err != nil {
			t.Fatalf("goinfer HiddenLast(%d tokens): %v", len(ids), err)
		}
		fastOut, prefillOut := make([]float32, fast.hidden), make([]float32, fast.hidden)
		if err := fast.HiddenLastInto(ids, fastOut, fastWS); err != nil {
			t.Fatalf("token-major HiddenLastInto(%d tokens): %v", len(ids), err)
		}
		if err := prefill.HiddenLastInto(ids, prefillOut, prefillWS); err != nil {
			t.Fatalf("layer-batched HiddenLastInto(%d tokens): %v", len(ids), err)
		}
		for name, got := range map[string][]float32{"token-major": fastOut, "layer-batched": prefillOut} {
			cos, maxAbs := vectorParity(got, want)
			t.Logf("%s tokens=%d cosine=%.9f max_abs=%.6g", name, len(ids), cos, maxAbs)
			if cos < 0.9999 || maxAbs > 0.3 {
				t.Fatalf("%s output diverged from goinfer: tokens=%d cosine=%.9f max_abs=%g", name, len(ids), cos, maxAbs)
			}
		}
		cos, maxAbs := vectorParity(prefillOut, fastOut)
		if cos < 0.99999 || maxAbs > 0.05 {
			t.Fatalf("layer-batched output diverged from token-major: tokens=%d cosine=%.9f max_abs=%g", len(ids), cos, maxAbs)
		}
	}
}

func BenchmarkPrefillOfficialTwelveTokens(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_FAST_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_FAST_MODEL")
	}
	model, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "int8", ExactPrefill: true})
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	fast, err := NewFastEvaluator(model)
	if err != nil {
		b.Fatal(err)
	}
	prefill := newPrefillEvaluator(fast)
	tok, err := tokenizer.Load(path)
	if err != nil {
		b.Fatal(err)
	}
	ids, err := tok.Encode("The Moon causes tides by pulling on Earth's oceans.", false)
	if err != nil {
		b.Fatal(err)
	}
	if len(ids) != 12 {
		b.Fatalf("official phrase encoded to %d tokens, want 12", len(ids))
	}
	ws, out := prefill.NewWorkspace(), make([]float32, fast.hidden)
	if err := prefill.HiddenLastInto(ids, out, ws); err != nil {
		b.Fatal(err)
	}
	if got := testing.AllocsPerRun(3, func() {
		if err := prefill.HiddenLastInto(ids, out, ws); err != nil {
			panic(err)
		}
	}); got != 0 {
		b.Fatalf("warmed official 12-token prefill allocated %.2f times/call, want zero", got)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := prefill.HiddenLastInto(ids, out, ws); err != nil {
			b.Fatal(err)
		}
	}
	benchmarkHiddenSink = out[0]
}
