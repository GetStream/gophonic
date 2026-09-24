// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"context"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/GetStream/gophonic/clm"
	"github.com/GetStream/gophonic/internal/q8gemm"
)

// Official-checkpoint gates. They need:
//
//	GOPHONIC_QWEN3_MODEL            official Qwen/Qwen3-8B safetensors snapshot
//	GOPHONIC_QWEN3_HELLO_REFERENCE  float32 last hidden state of "hello" from
//	                                tools/reference_hidden.py (BF16 PyTorch)
//	GOPHONIC_CLM_HEAD_BUNDLE        converted CLM v0.1 head
//
// GOPHONIC_QWEN_THREADS overrides the worker count for benchmarks.

var officialEncoders sync.Map // weight format -> *officialEncoder

type officialEncoder struct {
	once sync.Once
	enc  *Encoder
	load time.Duration
	err  error
}

func loadOfficialEncoder(tb testing.TB, format string) (*Encoder, time.Duration) {
	tb.Helper()
	path := os.Getenv("GOPHONIC_QWEN3_MODEL")
	if path == "" {
		tb.Skip("set GOPHONIC_QWEN3_MODEL to the official Qwen3-8B snapshot")
	}
	v, _ := officialEncoders.LoadOrStore(format, &officialEncoder{})
	o := v.(*officialEncoder)
	o.once.Do(func() {
		// Benchmarks repeat inputs; keep the cache out of compute timings.
		opts := Options{Weights: format, CacheEntries: -1}
		if n, err := strconv.Atoi(os.Getenv("GOPHONIC_QWEN_THREADS")); err == nil && n > 0 {
			opts.Threads = n
		}
		start := time.Now()
		o.enc, o.err = OpenWithOptions(path, opts)
		o.load = time.Since(start)
	})
	if o.err != nil {
		tb.Fatalf("load official Qwen3-8B (%s): %v", format, o.err)
	}
	return o.enc, o.load
}

func readReferenceVector(tb testing.TB, env string) []float32 {
	tb.Helper()
	path := os.Getenv(env)
	if path == "" {
		tb.Skipf("set %s", env)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		tb.Fatal(err)
	}
	if len(raw) != 4*hiddenSize {
		tb.Fatalf("%s holds %d bytes, want %d", path, len(raw), 4*hiddenSize)
	}
	out := make([]float32, hiddenSize)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), len(raw)), raw)
	return out
}

// TestOfficialHelloMatchesBF16Reference is the fidelity gate against the
// official BF16 PyTorch hidden state. For comparison, llama.cpp's Qwen3-8B
// Q8_0 GGUF reaches cosine 0.99929 on the same input.
func TestOfficialHelloMatchesBF16Reference(t *testing.T) {
	want := readReferenceVector(t, "GOPHONIC_QWEN3_HELLO_REFERENCE")
	for _, tc := range []struct {
		format string
		cosine float64
	}{
		{WeightsF16, 0.9999},
		{WeightsInt8, 0.997},
	} {
		enc, load := loadOfficialEncoder(t, tc.format)
		got := [][]float32{make([]float32, hiddenSize)}
		if err := enc.Embed(context.Background(), clm.StateRole, []string{"hello"}, got); err != nil {
			t.Fatal(err)
		}
		cos, maxAbs := vectorParity(got[0], want)
		t.Logf("%s: cosine vs official BF16 = %.6f (max_abs %.4g), load %s, projection bytes %.2f GiB",
			tc.format, cos, maxAbs, load.Round(time.Millisecond), float64(enc.model.WeightBytes())/(1<<30))
		if cos < tc.cosine {
			t.Errorf("%s: cosine %.6f below gate %.4f", tc.format, cos, tc.cosine)
		}
	}
}

func TestOfficialCLMRankingAndZeroAlloc(t *testing.T) {
	headPath := os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE")
	if headPath == "" {
		t.Skip("set GOPHONIC_CLM_HEAD_BUNDLE to the converted official CLM head")
	}
	head, err := clm.Load(headPath)
	if err != nil {
		t.Fatal(err)
	}
	state := "What causes tides on Earth?"
	candidates := []string{"The Moon’s gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."}
	wantNames := []string{"The Moon’s gravitational pull.", "Because the Earth is round.", "Photosynthesis in plants."}
	// Official Qwen3-8B BF16 hidden states through the PyTorch CLM heads.
	wantProb := []float64{0.9981802701950073, 0.0018110087839886546, 8.711908776604105e-06}
	for _, tc := range []struct {
		format    string
		tolerance float64
	}{
		{WeightsF16, 0.0002},
		{WeightsInt8, 0.0015},
	} {
		enc, _ := loadOfficialEncoder(t, tc.format)
		engine, err := clm.NewEngine(head, enc)
		if err != nil {
			t.Fatal(err)
		}
		ranks, err := engine.Rank(context.Background(), state, candidates, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i := range wantNames {
			if ranks[i].Candidate != wantNames[i] || ranks[i].Rank != i+1 {
				t.Fatalf("%s rank %d = %+v, want %q", tc.format, i+1, ranks[i], wantNames[i])
			}
			delta := math.Abs(float64(ranks[i].Probability) - wantProb[i])
			t.Logf("%s rank %d %q p=%.9f official-BF16=%.9f |Δ|=%.2g", tc.format, i+1, ranks[i].Candidate, ranks[i].Probability, wantProb[i], delta)
			if delta > tc.tolerance {
				t.Errorf("%s rank %d probability differs from official BF16 by %.3g", tc.format, i+1, delta)
			}
		}
		rankWS, err := engine.NewWorkspace(len(candidates))
		if err != nil {
			t.Fatal(err)
		}
		buf := make([]clm.RankedCandidate, len(candidates))
		rank := func() {
			if err := engine.RankInto(context.Background(), state, candidates, 1, buf, rankWS); err != nil {
				panic(err)
			}
		}
		rank()
		if allocs := testing.AllocsPerRun(2, rank); allocs != 0 {
			t.Fatalf("%s: warmed RankInto allocated %.2f times/call", tc.format, allocs)
		}
	}
}

// benchmarkTexts returns n texts of roughly tokens tokens each.
func benchmarkTexts(n, tokens int) []string {
	words := strings.Fields("the moon causes tides by pulling on earth's oceans while wind and pressure shape waves across the shallow coastal shelf every day")
	out := make([]string, n)
	for i := range out {
		var b strings.Builder
		for w := range tokens {
			if w > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(words[(w+i*7)%len(words)])
		}
		out[i] = b.String()
	}
	return out
}

// BenchmarkOfficialRankCached measures a CLM ranking whose state and
// candidates were embedded before, served by the exact embedding cache.
func BenchmarkOfficialRankCached(b *testing.B) {
	headPath := os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE")
	if headPath == "" {
		b.Skip("set GOPHONIC_CLM_HEAD_BUNDLE")
	}
	head, err := clm.Load(headPath)
	if err != nil {
		b.Fatal(err)
	}
	base, _ := loadOfficialEncoder(b, WeightsF16)
	enc, err := newEncoder(base.model, base.tokens, Options{}.threads(), defaultCacheEntries)
	if err != nil {
		b.Fatal(err)
	}
	defer enc.Close()
	engine, err := clm.NewEngine(head, enc)
	if err != nil {
		b.Fatal(err)
	}
	candidates := benchmarkTexts(16, 12)
	ws, err := engine.NewWorkspace(len(candidates))
	if err != nil {
		b.Fatal(err)
	}
	out := make([]clm.RankedCandidate, len(candidates))
	ctx := context.Background()
	state := "What causes tides on Earth?"
	start := time.Now()
	if err := engine.RankInto(ctx, state, candidates, 1, out, ws); err != nil {
		b.Fatal(err)
	}
	b.ReportMetric(float64(time.Since(start).Milliseconds()), "cold-ms")
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := engine.RankInto(ctx, state, candidates, 1, out, ws); err != nil {
			b.Fatal(err)
		}
	}
	hits, lookups := enc.CacheStats()
	b.Logf("cache hits %d of %d lookups", hits, lookups)
}

func BenchmarkOfficialEmbed(b *testing.B) {
	format := os.Getenv("GOPHONIC_QWEN_WEIGHTS")
	if format == "" {
		format = WeightsF16
	}
	enc, _ := loadOfficialEncoder(b, format)
	for _, tc := range []struct {
		name  string
		texts []string
	}{
		{"1text-1tok", []string{"hello"}},
		{"1text-12tok", []string{"The Moon causes tides by pulling on Earth's oceans."}},
		{"1text-64tok", benchmarkTexts(1, 64)},
		{"16texts-12tok", benchmarkTexts(16, 12)},
		{"1text-2048tok", benchmarkTexts(1, 1900)},
	} {
		b.Run(tc.name, func(b *testing.B) {
			dst := make([][]float32, len(tc.texts))
			for i := range dst {
				dst[i] = make([]float32, hiddenSize)
			}
			ctx := context.Background()
			if err := enc.Embed(ctx, clm.StateRole, tc.texts, dst); err != nil {
				b.Fatal(err)
			}
			tokens := 0
			for _, ids := range enc.tokenBufs[:len(tc.texts)] {
				tokens += len(ids)
			}
			retries := q8gemm.Retries()
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err := enc.Embed(ctx, clm.StateRole, tc.texts, dst); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			perOp := b.Elapsed().Seconds() / float64(b.N)
			b.ReportMetric(float64(tokens)/perOp, "tokens/s")
			b.ReportMetric(float64(q8gemm.Retries()-retries)/float64(b.N), "sme-retries/op")
			benchmarkHiddenSink = dst[0][0]
		})
	}
}
