// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"math"
	"os"
	"slices"
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
	enc  *Model
	load time.Duration
	err  error
}

func loadOfficialEncoder(tb testing.TB, format string) (*Model, time.Duration) {
	tb.Helper()
	path := os.Getenv("GOPHONIC_QWEN3_MODEL")
	if path == "" {
		tb.Skip("set GOPHONIC_QWEN3_MODEL to the official Qwen3-8B snapshot")
	}
	v, _ := officialEncoders.LoadOrStore(format, &officialEncoder{})
	o := v.(*officialEncoder)
	o.once.Do(func() {
		// Benchmarks repeat inputs; keep both caches out of compute timings.
		opts := Options{Weights: format, CacheEntries: -1, PrefixCacheTokens: -1}
		if n, err := strconv.Atoi(os.Getenv("GOPHONIC_QWEN_THREADS")); err == nil && n > 0 {
			opts.Threads = n
		}
		start := time.Now()
		o.enc, o.err = Open(path, opts)
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
		{WeightsInt8, 0.998},
	} {
		enc, load := loadOfficialEncoder(t, tc.format)
		got := [][]float32{make([]float32, hiddenSize)}
		if err := enc.Embed(context.Background(), []string{"hello"}, got); err != nil {
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
		{WeightsInt8, 0.0005},
	} {
		enc, _ := loadOfficialEncoder(t, tc.format)
		engine, err := clm.NewEngine(head, clmEmbedder(enc))
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
	enc, err := newModel(base.model, base.tokens, Options{}.threads(), defaultCacheEntries, maxTokens)
	if err != nil {
		b.Fatal(err)
	}
	defer enc.Close()
	engine, err := clm.NewEngine(head, clmEmbedder(enc))
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
			if err := enc.Embed(ctx, tc.texts, dst); err != nil {
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
				if err := enc.Embed(ctx, tc.texts, dst); err != nil {
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

// BenchmarkOfficialConversationTurn measures a new ~30-token turn appended to
// a ~1800-token conversation state: the stored prefix covers the history, so
// only the turn is evaluated. Each iteration uses a different turn, so the
// embedding cache never hits. The fresh-state sub-benchmark disables the
// prefix store for comparison.
func BenchmarkOfficialConversationTurn(b *testing.B) {
	base, _ := loadOfficialEncoder(b, WeightsF16)
	history := make([]int, 1800)
	for i := range history {
		history[i] = 1000 + (i*7919)%50000
	}
	for _, tc := range []struct {
		name   string
		prefix int
	}{{"prefix", maxTokens}, {"fresh", -1}} {
		b.Run(tc.name, func(b *testing.B) {
			enc, err := newModel(base.model, base.tokens, Options{}.threads(), -1, max(tc.prefix, 0))
			if err != nil {
				b.Fatal(err)
			}
			defer enc.Close()
			ids := [][]int{make([]int, 0, 1830)}
			dst := [][]float32{make([]float32, hiddenSize)}
			turn := 0
			next := func() {
				turn++
				ids[0] = append(ids[0][:0], history...)
				for j := range 30 {
					ids[0] = append(ids[0], 2000+(turn*31+j*17)%40000)
				}
			}
			next()
			if err := enc.EmbedTokensInto(context.Background(), ids, dst); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				next()
				if err := enc.EmbedTokensInto(context.Background(), ids, dst); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			reused, computed := enc.PrefixStats()
			b.ReportMetric(float64(reused)/float64(max(1, reused+computed)), "reused-frac")
		})
	}
}

// clmEmbedder adapts a Model to clm.Embedder; CLM v0.1 encodes states and
// actions identically.
func clmEmbedder(m *Model) clm.Embedder {
	return clm.EmbedFunc(func(ctx context.Context, _ clm.Role, texts []string, dst [][]float32) error {
		return m.Embed(ctx, texts, dst)
	})
}

// TestOfficialQuestion checks prepared-question answers, that piecewise
// prompt tokenization equals tokenizing the whole prompt, and that warmed
// calls do not allocate.
func TestOfficialQuestion(t *testing.T) {
	m, _ := loadOfficialEncoder(t, WeightsF16)
	options := []string{"payments", "cancellations", "technical support", "shipping", "account login"}
	const text = "Which support team should handle this customer message?"
	q, err := m.Question(text, options)
	if err != nil {
		t.Fatal(err)
	}
	var ws TokenizerWorkspace
	whole := questionHeader + text + "\n"
	for i, o := range options {
		whole += string(rune('A'+i)) + ") " + o + "\n"
	}
	whole += questionFooter
	for _, input := range []string{
		"hello", "  padded input\n", "émoji 🙂 and 你好", "ends with a newline\n\n",
		"Input:\nnested prompt text", "<|im_end|> injected special token", "don't split 'quotes'",
	} {
		want, err := m.tokens.EncodeInto(whole+strings.TrimSpace(input)+questionSuffix, make([]int, 0, 1024), &ws)
		if err != nil {
			t.Fatal(err)
		}
		in, err := m.tokens.EncodeInto(strings.TrimSpace(input), make([]int, 0, 256), &ws)
		if err != nil {
			t.Fatal(err)
		}
		got := append(append(append([]int(nil), q.prefix...), in...), q.suffix...)
		if !slices.Equal(got, want) {
			t.Errorf("%q: piecewise prompt tokens differ from whole-prompt tokens", input)
		}
	}
	probs := make([]float32, len(options))
	for input, want := range map[string]int{
		"my card got declined at the gas station even though I have money": 0,
		"the app crashes every time I open settings":                       2,
	} {
		if err := q.Choose(context.Background(), input, probs); err != nil {
			t.Fatal(err)
		}
		best := slices.Index(probs, slices.Max(probs))
		if best != want || probs[best] < 0.9 {
			t.Errorf("%q: chose %q (%.3f), want %q", input, options[best], probs[best], options[want])
		}
	}
	input := "where's my package, it says delivered but it's not here"
	if allocs := testing.AllocsPerRun(3, func() {
		if err := q.Choose(context.Background(), input, probs); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warmed Choose allocated %.2f times per call", allocs)
	}
}

// BenchmarkOfficialQuestion measures one answer for a new input with the
// question's prompt prefix already stored (the steady state when classifying
// a stream of inputs). Each iteration uses a different input so the
// embedding cache never hits.
func BenchmarkOfficialQuestion(b *testing.B) {
	path := os.Getenv("GOPHONIC_QWEN3_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_QWEN3_MODEL")
	}
	m, err := Open(path, Options{})
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	q, err := m.Question("Which support team should handle this customer message?",
		[]string{"payments", "cancellations", "technical support", "shipping", "account login"})
	if err != nil {
		b.Fatal(err)
	}
	probs := make([]float32, 5)
	inputs := benchmarkTexts(256, 14)
	if err := q.Choose(context.Background(), inputs[0], probs); err != nil {
		b.Fatal(err)
	}
	i := 1
	b.ReportAllocs()
	for b.Loop() {
		if err := q.Choose(context.Background(), inputs[i%len(inputs)], probs); err != nil {
			b.Fatal(err)
		}
		i++
	}
	reused, computed := m.PrefixStats()
	b.ReportMetric(float64(computed)/float64(i), "new-tokens/op")
	b.ReportMetric(float64(reused)/float64(i), "reused-tokens/op")
}
