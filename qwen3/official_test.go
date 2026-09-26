// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/GetStream/gophonic/clm"
	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
	"github.com/GetStream/gophonic/internal/testmodels"
)

// Official-checkpoint gates run when the models directory (see
// internal/testmodels) holds the Qwen3-8B snapshot, and, for some, the BF16
// "hello" reference vector or the converted CLM head. Benchmarks run one
// sub-benchmark per weight format: select one with -bench 'Name/gpu'.

var officialEncoders sync.Map // weight format -> *officialEncoder

type officialEncoder struct {
	once sync.Once
	enc  *Model
	load time.Duration
	err  error
}

// officialFormats are the weight formats the official benchmarks compare.
var officialFormats = []string{WeightsF16, WeightsInt8, WeightsGPU, WeightsGPUQ4}

// forEachFormat runs bench as one sub-benchmark per weight format.
func forEachFormat(b *testing.B, bench func(b *testing.B, format string)) {
	for _, format := range officialFormats {
		b.Run(format, func(b *testing.B) { bench(b, format) })
	}
}

func loadOfficialEncoder(tb testing.TB, format string) (*Model, time.Duration) {
	tb.Helper()
	path := testmodels.Path(tb, testmodels.Qwen3)
	v, _ := officialEncoders.LoadOrStore(format, &officialEncoder{})
	o := v.(*officialEncoder)
	if (format == WeightsGPU || format == WeightsGPUQ4) && (runtime.GOOS != "darwin" || runtime.GOARCH != "arm64") {
		tb.Skipf("%s needs the Apple GPU", format)
	}
	o.once.Do(func() {
		// Benchmarks repeat inputs; keep both caches out of compute timings.
		opts := Options{Weights: format, CacheEntries: -1, PrefixCacheTokens: -1}
		start := time.Now()
		o.enc, o.err = Open(path, opts)
		o.load = time.Since(start)
	})
	if o.err != nil {
		tb.Fatalf("load official Qwen3-8B (%s): %v", format, o.err)
	}
	return o.enc, o.load
}

// hiddenSize is the width of Qwen3-8B, the model the official tests load.
const hiddenSize = 4096

func readReferenceVector(tb testing.TB, name string) []float32 {
	tb.Helper()
	path := testmodels.Path(tb, name)
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
	want := readReferenceVector(t, testmodels.Qwen3HelloReference)
	for _, tc := range []struct {
		format string
		cosine float64
	}{
		{WeightsF16, 0.9999},
		{WeightsInt8, 0.998},
		{WeightsGPU, 0.999},
		{WeightsGPUQ4, 0.95}, // llama.cpp Q4_K_M: 0.942
	} {
		if (tc.format == WeightsGPU || tc.format == WeightsGPUQ4) && (runtime.GOOS != "darwin" || runtime.GOARCH != "arm64") {
			continue
		}
		enc, load := loadOfficialEncoder(t, tc.format)
		got := [][]float32{make([]float32, hiddenSize)}
		if err := enc.Embed(context.Background(), []string{"hello"}, got); err != nil {
			t.Fatal(err)
		}
		cos, maxAbs := lmtest.VectorParity(got[0], want)
		t.Logf("%s: cosine vs official BF16 = %.6f (max_abs %.4g), load %s, projection bytes %.2f GiB",
			tc.format, cos, maxAbs, load.Round(time.Millisecond), float64(enc.model.WeightBytes())/(1<<30))
		if cos < tc.cosine {
			t.Errorf("%s: cosine %.6f below gate %.4f", tc.format, cos, tc.cosine)
		}
	}
}

func TestOfficialCLMRankingAndZeroAlloc(t *testing.T) {
	headPath := testmodels.Path(t, testmodels.CLMHead)
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
	headPath := testmodels.Path(b, testmodels.CLMHead)
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

func BenchmarkOfficialEmbed(b *testing.B) { forEachFormat(b, benchmarkEmbed) }

func benchmarkEmbed(b *testing.B, format string) {
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
func BenchmarkOfficialConversationTurn(b *testing.B) { forEachFormat(b, benchmarkConversationTurn) }

func benchmarkConversationTurn(b *testing.B, format string) {
	base, _ := loadOfficialEncoder(b, format)
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
		want, err := m.tokens.EncodeInto(whole+strings.TrimSpace(input)+questionSuffix+m.answer, make([]int, 0, 1024), &ws)
		if err != nil {
			t.Fatal(err)
		}
		in, err := m.tokens.EncodeInto(strings.TrimSpace(input), make([]int, 0, 256), &ws)
		if err != nil {
			t.Fatal(err)
		}
		got := append(append(append([]int(nil), q.kv.Tokens()...), in...), q.suffix...)
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

	// A second question interleaved with the first must not disturb it,
	// and a batch must agree with one-at-a-time answers.
	q2, err := m.Question("Is this message positive or negative?", []string{"positive", "negative"})
	if err != nil {
		t.Fatal(err)
	}
	batch := []string{
		"my card got declined at the gas station even though I have money",
		"cancel my plan before it renews next week",
		"the app crashes every time I open settings",
		"where's my package, it says delivered but it's not here",
		"I forgot my password and the reset email never arrives",
	}
	rows := make([][]float32, len(batch))
	for i := range rows {
		rows[i] = make([]float32, len(options))
	}
	if err := q.ChooseBatch(context.Background(), batch, rows); err != nil {
		t.Fatal(err)
	}
	two := make([]float32, 2)
	for i, in := range batch {
		if err := q2.Choose(context.Background(), "I love this!", two); err != nil {
			t.Fatal(err)
		}
		if two[0] < 0.9 {
			t.Errorf("second question: P(positive)=%.3f", two[0])
		}
		if err := q.Choose(context.Background(), in, probs); err != nil {
			t.Fatal(err)
		}
		if best := slices.Index(rows[i], slices.Max(rows[i])); best != i {
			t.Errorf("batch input %d chose %q", i, options[best])
		}
		for j := range probs {
			if d := math.Abs(float64(probs[j] - rows[i][j])); d > 1e-3 {
				t.Errorf("input %d option %d: single %.5f, batch %.5f", i, j, probs[j], rows[i][j])
			}
		}
	}
	if allocs := testing.AllocsPerRun(2, func() {
		if err := q.ChooseBatch(context.Background(), batch, rows); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warmed ChooseBatch allocated %.2f times per call", allocs)
	}
}

// BenchmarkOfficialChooseBatch measures answering 16 new inputs per call
// against one prepared question.
func BenchmarkOfficialChooseBatch(b *testing.B) {
	for _, format := range officialFormats {
		b.Run(format, func(b *testing.B) {
			m, _ := loadOfficialEncoder(b, format)
			q, err := m.Question("Which support team should handle this customer message?",
				[]string{"payments", "cancellations", "technical support", "shipping", "account login"})
			if err != nil {
				b.Fatal(err)
			}
			const n = 16
			rows := make([][]float32, n)
			for i := range rows {
				rows[i] = make([]float32, 5)
			}
			texts := benchmarkTexts(n*64, 14)
			if err := q.ChooseBatch(context.Background(), texts[:n], rows); err != nil {
				b.Fatal(err)
			}
			i := 1
			b.ReportAllocs()
			for b.Loop() {
				start := (i * n) % (len(texts) - n)
				if err := q.ChooseBatch(context.Background(), texts[start:start+n], rows); err != nil {
					b.Fatal(err)
				}
				i++
			}
			b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N)/n, "ms/input")
		})
	}
}

// BenchmarkOfficialQuestion measures one answer for a new input with the
// question's prompt prefix already stored (the steady state when classifying
// a stream of inputs). Each iteration uses a different input so the
// embedding cache never hits.
func BenchmarkOfficialQuestion(b *testing.B) { forEachFormat(b, benchmarkQuestion) }

func benchmarkQuestion(b *testing.B, format string) {
	m, _ := loadOfficialEncoder(b, format)
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
}

// TestOfficialStream feeds growing and revised partial transcripts to a
// Stream and checks each answer against a fresh Choose on the same text.
func TestOfficialStream(t *testing.T) {
	m, _ := loadOfficialEncoder(t, WeightsF16)
	q, err := m.Question("A voice assistant hears this live, unpunctuated transcript. Has the user finished their turn, or did they stop mid-sentence and will keep talking?",
		[]string{"reply now", "wait"})
	if err != nil {
		t.Fatal(err)
	}
	s, err := q.NewStream(256)
	if err != nil {
		t.Fatal(err)
	}
	partials := []string{
		"can you", "can you book", "can you book me a table", "can you book me a table for two at",
		"can you book me a table for two at seven tonight",
		"can you book me a table for three at eight tonight", // revision
		"so i was thinking maybe we could",
	}
	got, want := make([]float32, 2), make([]float32, 2)
	for _, text := range partials {
		if err := s.Update(context.Background(), text, got); err != nil {
			t.Fatal(err)
		}
		if err := q.Choose(context.Background(), text, want); err != nil {
			t.Fatal(err)
		}
		if d := math.Abs(float64(got[0] - want[0])); d > 2e-3 {
			t.Errorf("%q: stream P(reply)=%.4f, fresh %.4f", text, got[0], want[0])
		}
		t.Logf("P(reply)=%.3f  %q", got[0], text)
	}
	text := partials[4]
	if allocs := testing.AllocsPerRun(3, func() {
		if err := s.Update(context.Background(), text, got); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warmed Update allocated %.2f times per call", allocs)
	}
}

// BenchmarkOfficialStream measures one stream update that adds three words
// to a 20-word partial transcript.
func BenchmarkOfficialStream(b *testing.B) { forEachFormat(b, benchmarkStream) }

func benchmarkStream(b *testing.B, format string) {
	m, _ := loadOfficialEncoder(b, format)
	q, err := m.Question("A voice assistant hears this live, unpunctuated transcript. Has the user finished their turn, or did they stop mid-sentence and will keep talking?",
		[]string{"reply now", "wait"})
	if err != nil {
		b.Fatal(err)
	}
	s, err := q.NewStream(256)
	if err != nil {
		b.Fatal(err)
	}
	base := benchmarkTexts(1, 20)[0]
	probs := make([]float32, 2)
	i := 0
	for b.Loop() {
		// Alternate between the base partial and one with three more words,
		// so every update evaluates about three new tokens plus the suffix.
		text := base
		if i%2 == 1 {
			text = base + " while the tide turns"
		}
		if err := s.Update(context.Background(), text, probs); err != nil {
			b.Fatal(err)
		}
		i++
	}
}

var contextConversation = []string{
	"Customer: Hi, I was charged twice for my order #4411 last week.",
	"Agent: I'm sorry about that. Let me look into the duplicate charge for you.",
	"Customer: It's been a week and nobody has answered my emails. This is really frustrating.",
	"Agent: I understand. I've now refunded the duplicate payment; it will reach your card in 3-5 business days.",
	"Customer: Okay, thanks. That's all I needed.",
}

func contextQuestions(tb testing.TB, m *Model) ([]*ContextQuestion, [][]string) {
	specs := []struct {
		q    string
		opts []string
	}{
		{"How does the customer feel at the end of this conversation?", []string{"satisfied", "angry", "confused"}},
		{"What was the customer's problem about?", []string{"billing", "shipping", "technical issue", "account login"}},
		{"Was the customer's issue resolved?", []string{"yes", "no"}},
	}
	qs := make([]*ContextQuestion, len(specs))
	opts := make([][]string, len(specs))
	for i, sp := range specs {
		q, err := m.ContextQuestion(sp.q, sp.opts)
		if err != nil {
			tb.Fatal(err)
		}
		qs[i], opts[i] = q, sp.opts
	}
	return qs, opts
}

// TestOfficialContext asks several questions about a growing conversation
// and checks tokenization, answers, incremental updates, and allocations.
func TestOfficialContext(t *testing.T) {
	m, _ := loadOfficialEncoder(t, WeightsF16)
	c, err := m.NewContext(1024)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Join(contextConversation, "\n")
	var ws TokenizerWorkspace
	whole, err := m.tokens.EncodeInto("<|im_start|>user\n"+text+contextSeparator, make([]int, 0, 1024), &ws)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Set(context.Background(), text); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(c.kv.Tokens(), whole) {
		t.Fatal("piecewise context tokens differ from whole-prompt tokens")
	}
	qs, opts := contextQuestions(t, m)
	full, err := m.tokens.EncodeInto("<|im_start|>user\n"+text+contextSeparator+
		"How does the customer feel at the end of this conversation?\nA) satisfied\nB) angry\nC) confused\n"+contextFooter+m.answer, make([]int, 0, 1024), &ws)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(append(slices.Clone(c.kv.Tokens()), qs[0].tail...), full) {
		t.Fatal("context plus question tail tokens differ from whole-prompt tokens")
	}
	probs := [][]float32{make([]float32, 3), make([]float32, 4), make([]float32, 2)}
	if err := c.Ask(context.Background(), qs, probs); err != nil {
		t.Fatal(err)
	}
	want := []string{"satisfied", "billing", "yes"}
	for i, p := range probs {
		best := slices.Index(p, slices.Max(p))
		t.Logf("%-10s %.3f", opts[i][best], p[best])
		if opts[i][best] != want[i] {
			t.Errorf("question %d answered %q, want %q", i, opts[i][best], want[i])
		}
	}
	// Growing the conversation evaluates only the new turn.
	r0, c0 := m.PrefixStats()
	_ = r0
	longer := text + "\nCustomer: Actually, wait, the refund never arrived and now I'm really angry."
	if err := c.Set(context.Background(), longer); err != nil {
		t.Fatal(err)
	}
	_ = c0
	if err := c.Ask(context.Background(), qs, probs); err != nil {
		t.Fatal(err)
	}
	if best := slices.Index(probs[0], slices.Max(probs[0])); opts[0][best] != "angry" {
		t.Errorf("after the new turn the customer feels %q, want angry", opts[0][best])
	}
	if allocs := testing.AllocsPerRun(3, func() {
		if err := c.Set(context.Background(), longer); err != nil {
			panic(err)
		}
		if err := c.Ask(context.Background(), qs, probs); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warmed Set+Ask allocated %.2f times per call", allocs)
	}
}

// BenchmarkOfficialContext compares three questions about a conversation
// asked with Context (context evaluated once) against three Question.Choose
// calls on the whole conversation.
func BenchmarkOfficialContext(b *testing.B) { forEachFormat(b, benchmarkContext) }

func benchmarkContext(b *testing.B, format string) {
	m, _ := loadOfficialEncoder(b, format)
	text := strings.Join(contextConversation, "\n")
	qs, _ := contextQuestions(b, m)
	probs := [][]float32{make([]float32, 3), make([]float32, 4), make([]float32, 2)}
	b.Run("context-ask", func(b *testing.B) {
		c, err := m.NewContext(1024)
		if err != nil {
			b.Fatal(err)
		}
		if err := c.Set(context.Background(), text); err != nil {
			b.Fatal(err)
		}
		for b.Loop() {
			if err := c.Ask(context.Background(), qs, probs); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("context-new-turn-and-ask", func(b *testing.B) {
		c, err := m.NewContext(1024)
		if err != nil {
			b.Fatal(err)
		}
		i := 0
		for b.Loop() {
			turn := fmt.Sprintf("%s\nCustomer: one more question number %d about my refund.", text, i)
			i++
			b.StopTimer()
			if err := c.Set(context.Background(), text); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
			if err := c.Set(context.Background(), turn); err != nil {
				b.Fatal(err)
			}
			if err := c.Ask(context.Background(), qs, probs); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("three-questions", func(b *testing.B) {
		specs := []struct {
			q    string
			opts []string
		}{
			{"How does the customer feel at the end of this conversation?", []string{"satisfied", "angry", "confused"}},
			{"What was the customer's problem about?", []string{"billing", "shipping", "technical issue", "account login"}},
			{"Was the customer's issue resolved?", []string{"yes", "no"}},
		}
		var qq []*Question
		for _, sp := range specs {
			q, err := m.Question(sp.q, sp.opts)
			if err != nil {
				b.Fatal(err)
			}
			qq = append(qq, q)
		}
		i := 0
		for b.Loop() {
			in := fmt.Sprintf("%s\nCustomer: one more question number %d about my refund.", text, i)
			i++
			for j, q := range qq {
				if err := q.Choose(context.Background(), in, probs[j]); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}
