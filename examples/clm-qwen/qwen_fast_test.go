// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/GetStream/gophonic/clm"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestFastWorkspaceNoSteadyStateAllocs(t *testing.T) {
	f, err := tinyFastEvaluator()
	if err != nil {
		t.Fatal(err)
	}
	ws := f.NewWorkspace()
	ids := []int{1, 2, 3}
	out := make([]float32, f.hidden)
	encoder := &Encoder{fast: f, ws: ws}
	sequences, dst := [][]int{ids}, [][]float32{out}
	if err := encoder.EmbedTokensInto(context.Background(), clm.StateRole, sequences, dst); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(100, func() {
		if err := encoder.EmbedTokensInto(context.Background(), clm.StateRole, sequences, dst); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("pretokenized int8 EmbedTokensInto allocated %.2f times/call after warmup, want 0", allocs)
	}
}

func TestFastQwenOfficialParity(t *testing.T) {
	path := os.Getenv("GOPHONIC_QWEN3_FAST_MODEL")
	if path == "" {
		t.Skip("set GOPHONIC_QWEN3_FAST_MODEL to the local Qwen3-8B checkpoint for the int8 parity gate")
	}
	model, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "int8", ExactPrefill: true})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	f, err := NewFastEvaluator(model)
	if err != nil {
		t.Fatal(err)
	}
	ws := f.NewWorkspace()
	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	one, err := tok.Encode("hello", false)
	if err != nil {
		t.Fatal(err)
	}
	many, err := tok.Encode("The Moon causes tides by pulling on Earth's oceans.", false)
	if err != nil {
		t.Fatal(err)
	}
	for _, ids := range [][]int{one, many} {
		want, err := model.HiddenLast(ids)
		if err != nil {
			t.Fatalf("goinfer HiddenLast(%v): %v", ids, err)
		}
		got := make([]float32, f.hidden)
		if err := f.HiddenLastInto(ids, got, ws); err != nil {
			t.Fatalf("fast HiddenLastInto(%v): %v", ids, err)
		}
		cos, maxAbs := vectorParity(got, want)
		t.Logf("tokens=%d cosine=%.9f max_abs=%.6g", len(ids), cos, maxAbs)
		if cos < 0.9999 || maxAbs > 0.25 {
			t.Fatalf("fast Qwen hidden diverged from goinfer: tokens=%d cosine=%.9f max_abs=%g", len(ids), cos, maxAbs)
		}
	}
	oneToken := one
	out := make([]float32, f.hidden)
	allocs := testing.AllocsPerRun(3, func() {
		if err := f.HiddenLastInto(oneToken, out, ws); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("official-checkpoint HiddenLastInto allocated %.2f times/call after warmup, want 0", allocs)
	}
}

func tinyFastEvaluator() (*FastEvaluator, error) {
	const hidden, vocab, intermediate = 8, 4, 12
	cfg := &decoder.Config{
		ModelType: "qwen3", VocabSize: vocab, HiddenDim: hidden, NumLayers: 1,
		NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: intermediate,
		MaxPositions: 16, RMSNormEps: 1e-6, RoPEGlobalBase: 1e6,
	}
	mat := func(rows, cols int, seed float32) linalg.WeightMat {
		data := make([]float32, rows*cols)
		for i := range data {
			data[i] = seed * float32((i%7)-3) / 10
		}
		return linalg.WrapF32(data, rows, cols)
	}
	norm := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}
	weights := &decoder.Weights{
		Cfg: *cfg, Embed: mat(vocab, hidden, 0.3), FinalNorm: norm(hidden),
		Layers: []decoder.LayerWeights{{
			QProj: mat(hidden, hidden, 0.4), KProj: mat(4, hidden, 0.5),
			VProj: mat(4, hidden, 0.6), OProj: mat(hidden, hidden, 0.7),
			GateProj: mat(intermediate, hidden, 0.8), UpProj: mat(intermediate, hidden, 0.9),
			DownProj: mat(hidden, intermediate, 1.0), PreAttnNorm: norm(hidden),
			PreMLPNorm: norm(hidden), QNorm: norm(4), KNorm: norm(4),
		}},
	}
	return newFastEvaluator(cfg, weights)
}

func vectorParity(a, b []float32) (cosine, maxAbs float64) {
	var dot, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		aa += x * x
		bb += y * y
		maxAbs = math.Max(maxAbs, math.Abs(x-y))
	}
	return dot / math.Sqrt(aa*bb), maxAbs
}
