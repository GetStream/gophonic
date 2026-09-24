// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"context"
	"math"
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
	"time"

	"github.com/GetStream/gophonic/clm"
	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const officialSMEText = "The Moon causes tides by pulling on Earth's oceans."

type officialSMEFixture struct {
	encoder    *Encoder
	ids        []int
	texts      []string
	dest       [][]float32
	packTime   time.Duration
	packedByte uint64
	heapBytes  uint64
}

func loadOfficialSMEFixture(tb testing.TB) *officialSMEFixture {
	tb.Helper()
	path := os.Getenv("GOPHONIC_QWEN3_FAST_MODEL")
	if path == "" {
		tb.Skip("set GOPHONIC_QWEN3_FAST_MODEL to the official Qwen3-8B directory")
	}
	if !q8gemm.Available() {
		tb.Skip("this CPU does not expose the 512-bit SME kernel")
	}

	model, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "int8", ExactPrefill: true})
	if err != nil {
		tb.Fatalf("load official Qwen3-8B: %v", err)
	}
	fast, err := NewFastEvaluator(model)
	if err != nil {
		_ = model.Close()
		tb.Fatalf("initialize fast evaluator: %v", err)
	}
	start := time.Now()
	prefill := newPrefillEvaluator(fast)
	packTime := time.Since(start)
	if prefill.setupErr != nil {
		_ = model.Close()
		tb.Fatalf("initialize SME prefill evaluator: %v", prefill.setupErr)
	}
	if !prefill.useQ8GEMM {
		_ = model.Close()
		tb.Fatal("q8gemm.Available is true but prefill did not enable the SME path")
	}
	packedBytes := uint64(0)
	for i := range prefill.packed {
		weights := [...]*q8gemm.Weights{
			prefill.packed[i].q, prefill.packed[i].k, prefill.packed[i].v, prefill.packed[i].o,
			prefill.packed[i].gate, prefill.packed[i].up, prefill.packed[i].down,
		}
		for j, weight := range weights {
			if weight == nil {
				_ = model.Close()
				tb.Fatalf("layer %d projection %d did not receive SME-packed weights", i, j)
			}
			packedBytes += uint64(weight.Bytes())
		}
	}
	if packedBytes == 0 {
		_ = model.Close()
		tb.Fatal("SME prefill has no packed Q8 projection weights")
	}

	baseTokenizer, err := tokenizer.Load(path)
	if err != nil {
		_ = model.Close()
		tb.Fatalf("load reference tokenizer: %v", err)
	}
	ids, err := baseTokenizer.Encode(officialSMEText, false)
	if err != nil {
		_ = model.Close()
		tb.Fatalf("tokenize official phrase: %v", err)
	}
	if len(ids) != 12 {
		_ = model.Close()
		tb.Fatalf("official phrase encoded to %d tokens, want 12", len(ids))
	}
	textTokenizer, err := LoadQwenTokenizer(path)
	if err != nil {
		_ = model.Close()
		tb.Fatalf("load allocation-free tokenizer: %v", err)
	}
	enc := &Encoder{
		model:      model,
		qwenTokens: textTokenizer,
		tokenIDs:   make([]int, 0, maxTokens),
		fast:       fast,
		ws:         fast.NewWorkspace(),
		prefill:    prefill,
		prefillWS:  prefill.NewWorkspace(),
	}
	warmText := []string{officialSMEText}
	warmDst := [][]float32{make([]float32, hiddenSize)}
	if err := enc.Embed(context.Background(), clm.StateRole, warmText, warmDst); err != nil {
		_ = enc.Close()
		tb.Fatalf("warm public SME inference: %v", err)
	}
	runtime.GC()
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	tb.Cleanup(func() { _ = enc.Close() })
	return &officialSMEFixture{
		encoder:    enc,
		ids:        ids,
		texts:      []string{officialSMEText},
		dest:       [][]float32{make([]float32, hiddenSize)},
		packTime:   packTime,
		packedByte: packedBytes,
		heapBytes:  mem.HeapAlloc,
	}
}

func TestOfficialSMEPrefillParityAndPublicZeroAlloc(t *testing.T) {
	fx := loadOfficialSMEFixture(t)
	ref, err := fx.encoder.model.HiddenLast(fx.ids)
	if err != nil {
		t.Fatalf("goinfer reference: %v", err)
	}
	if err := fx.encoder.Embed(context.Background(), clm.StateRole, fx.texts, fx.dest); err != nil {
		t.Fatalf("public text Embed: %v", err)
	}
	cos, maxAbs, rms := hiddenDiff(fx.dest[0], ref)
	t.Logf("12-token SME vs goinfer hidden: cosine=%.9f max_abs=%.6g rms=%.6g pack=%s packed_projection_bytes=%d", cos, maxAbs, rms, fx.packTime, fx.packedByte)
	if cos < 0.99999 || maxAbs > 0.001 || rms > 0.0001 {
		t.Fatalf("SME prefill diverged from goinfer reference: cosine=%.9f max_abs=%g rms=%g", cos, maxAbs, rms)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if err := fx.encoder.Embed(context.Background(), clm.StateRole, fx.texts, fx.dest); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warmed public text Embed allocated %.2f times/call, want zero", allocs)
	}
}

func TestOfficialSMECLMRankingParity(t *testing.T) {
	headPath := os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE")
	if headPath == "" {
		t.Skip("set GOPHONIC_CLM_HEAD_BUNDLE to the converted official CLM head")
	}
	head, err := clm.Load(headPath)
	if err != nil {
		t.Fatalf("load official CLM head: %v", err)
	}
	fx := loadOfficialSMEFixture(t)
	state := "What causes tides on Earth?"
	candidates := []string{
		"The Moon’s gravitational pull.",
		"Photosynthesis in plants.",
		"Because the Earth is round.",
	}
	oldEncoder := &Encoder{
		qwenTokens: fx.encoder.qwenTokens,
		tokenIDs:   make([]int, 0, maxTokens),
		fast:       fx.encoder.fast,
		ws:         fx.encoder.fast.NewWorkspace(),
	}
	oldEngine, err := clm.NewEngine(head, oldEncoder)
	if err != nil {
		t.Fatal(err)
	}
	smeEngine, err := clm.NewEngine(head, fx.encoder)
	if err != nil {
		t.Fatal(err)
	}
	oldRanks, err := oldEngine.Rank(context.Background(), state, candidates, 1)
	if err != nil {
		t.Fatalf("old int8 token-major rank: %v", err)
	}
	smeRanks, err := smeEngine.Rank(context.Background(), state, candidates, 1)
	if err != nil {
		t.Fatalf("SME layer-batched rank: %v", err)
	}
	wantNames := []string{
		"The Moon’s gravitational pull.",
		"Because the Earth is round.",
		"Photosynthesis in plants.",
	}
	wantProb := []float64{0.9981802701950073, 0.0018110087839886546, 8.711908776604105e-06}
	for i := range wantNames {
		if smeRanks[i].Candidate != wantNames[i] || smeRanks[i].Rank != i+1 {
			t.Fatalf("SME rank %d = %+v, want %q", i+1, smeRanks[i], wantNames[i])
		}
		if oldRanks[i].Candidate != wantNames[i] || oldRanks[i].Rank != i+1 {
			t.Fatalf("old int8 rank %d = %+v, want %q", i+1, oldRanks[i], wantNames[i])
		}
		if delta := math.Abs(float64(smeRanks[i].Probability - oldRanks[i].Probability)); delta > 0.0001 {
			t.Errorf("rank %d SME probability %.8f differs from old int8 %.8f by %.6g", i+1, smeRanks[i].Probability, oldRanks[i].Probability, delta)
		}
		if delta := math.Abs(float64(smeRanks[i].Probability) - wantProb[i]); delta > 0.0012 {
			t.Errorf("rank %d SME probability %.8f differs from official BF16 %.8f by %.6g", i+1, smeRanks[i].Probability, wantProb[i], delta)
		}
		t.Logf("rank %d %q SME=%.9f old-int8=%.9f official-BF16=%.9f", i+1, smeRanks[i].Candidate, smeRanks[i].Probability, oldRanks[i].Probability, wantProb[i])
	}
	rankWS, err := smeEngine.NewWorkspace(len(candidates))
	if err != nil {
		t.Fatal(err)
	}
	rankBuffer := make([]clm.RankedCandidate, len(candidates))
	rank := func() {
		if err := smeEngine.RankInto(context.Background(), state, candidates, 1, rankBuffer, rankWS); err != nil {
			panic(err)
		}
	}
	rank()
	if allocs := testing.AllocsPerRun(1, rank); allocs != 0 {
		t.Fatalf("warmed public RankInto allocated %.2f times/call, want zero", allocs)
	}
}

func BenchmarkOfficialQwenSMEPublicTwelveTokens(b *testing.B) {
	fx := loadOfficialSMEFixture(b)
	if err := fx.encoder.Embed(context.Background(), clm.StateRole, fx.texts, fx.dest); err != nil {
		b.Fatal(err)
	}
	if got := testing.AllocsPerRun(5, func() {
		if err := fx.encoder.Embed(context.Background(), clm.StateRole, fx.texts, fx.dest); err != nil {
			panic(err)
		}
	}); got != 0 {
		b.Fatalf("warmed public text Embed allocated %.2f times/call, want zero", got)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var profileFile *os.File
	if profilePath := os.Getenv("GOPHONIC_QWEN_SME_CPU_PROFILE"); profilePath != "" {
		var err error
		profileFile, err = os.Create(profilePath)
		if err != nil {
			b.Fatal(err)
		}
		if err := pprof.StartCPUProfile(profileFile); err != nil {
			_ = profileFile.Close()
			b.Fatal(err)
		}
	}
	collectSamples := os.Getenv("GOPHONIC_QWEN_SME_SAMPLE_TIMES") == "1"
	var samples [256]time.Duration
	sampleCount := 0
	for b.Loop() {
		start := time.Time{}
		if collectSamples {
			start = time.Now()
		}
		if err := fx.encoder.Embed(context.Background(), clm.StateRole, fx.texts, fx.dest); err != nil {
			b.Fatal(err)
		}
		if collectSamples && sampleCount < len(samples) {
			samples[sampleCount] = time.Since(start)
			sampleCount++
		}
	}
	benchmarkHiddenSink = fx.dest[0][0]
	if profileFile != nil {
		pprof.StopCPUProfile()
		if err := profileFile.Close(); err != nil {
			b.Fatal(err)
		}
	}
	if collectSamples {
		b.StopTimer()
		for i, duration := range samples[:sampleCount] {
			b.Logf("public_embed_sample_%02d=%s", i+1, duration)
		}
	}
	b.ReportMetric(fx.packTime.Seconds(), "pack-s")
	b.ReportMetric(float64(fx.packedByte)/(1024*1024), "repack-MiB")
	b.ReportMetric(float64(fx.heapBytes)/(1024*1024*1024), "heap-GiB")
}

func hiddenDiff(got, want []float32) (cosine, maxAbs, rms float64) {
	if len(got) != len(want) || len(got) == 0 {
		return 0, math.Inf(1), math.Inf(1)
	}
	var dot, gotNorm, wantNorm, squaredError float64
	for i, value := range got {
		reference := float64(want[i])
		delta := float64(value) - reference
		dot += float64(value) * reference
		gotNorm += float64(value) * float64(value)
		wantNorm += reference * reference
		squaredError += delta * delta
		if abs := math.Abs(delta); abs > maxAbs {
			maxAbs = abs
		}
	}
	cosine = dot / math.Sqrt(gotNorm*wantNorm)
	rms = math.Sqrt(squaredError / float64(len(got)))
	return cosine, maxAbs, rms
}
