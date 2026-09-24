// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/gophonic/clm"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

func TestPackedQ8CompactionAndChunkedPrefillParity(t *testing.T) {
	fast, err := tinyInt8FastEvaluator()
	if err != nil {
		t.Fatal(err)
	}
	prefill := &PrefillEvaluator{fast: fast, useQ8GEMM: true}
	prefill.packed, err = packQ8Layers(fast.w)
	if err != nil {
		t.Fatal(err)
	}
	fastWS := fast.NewWorkspace()
	idsByLength := make([][]int, 0, 4)
	wantByLength := make([][]float32, 0, 4)
	for _, n := range []int{1, 12, 17, 128} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*3 + 1) % fast.cfg.VocabSize
		}
		want := make([]float32, fast.hidden)
		if err := fast.HiddenLastInto(ids, want, fastWS); err != nil {
			t.Fatalf("reference %d-token fast path: %v", n, err)
		}
		idsByLength = append(idsByLength, ids)
		wantByLength = append(wantByLength, want)
	}

	incomplete := append([]packedQ8Layer(nil), prefill.packed...)
	incomplete[0].v = nil
	if compactQ8ProjectionWeights(fast.w, incomplete) {
		t.Fatal("compaction accepted an incomplete packed projection set")
	}
	for _, mat := range layerProjectionMats(&fast.w.Layers[0]) {
		if mat.Kind() != "int8" {
			t.Fatalf("failed compaction partially dropped a projection: kind=%q", mat.Kind())
		}
	}
	if !prefill.compactOwnedWeights() {
		t.Fatal("complete packed projection set was not compacted")
	}
	if !prefill.weightsCompacted {
		t.Fatal("compaction invariant was not recorded")
	}
	for layer := range fast.w.Layers {
		for _, mat := range layerProjectionMats(&fast.w.Layers[layer]) {
			if mat.Kind() != "" {
				t.Fatalf("layer %d canonical projection remains: %q", layer, mat.Kind())
			}
		}
	}

	prefillWS := prefill.NewWorkspace()
	for i, ids := range idsByLength {
		got := make([]float32, fast.hidden)
		if err := prefill.HiddenLastInto(ids, got, prefillWS); err != nil {
			t.Fatalf("compacted %d-token prefill: %v", len(ids), err)
		}
		cos, maxAbs := vectorParity(got, wantByLength[i])
		if cos < 0.99999 || maxAbs > 1e-4 {
			t.Fatalf("compacted %d-token parity failed: cosine=%.9f max_abs=%g", len(ids), cos, maxAbs)
		}
		if got := testing.AllocsPerRun(20, func() {
			if err := prefill.HiddenLastInto(ids, got, prefillWS); err != nil {
				panic(err)
			}
		}); got != 0 {
			t.Fatalf("compacted %d-token prefill allocated %.2f times/call", len(ids), got)
		}
	}
}

func TestCompactQ8ProjectionWeightsRequiresAllQwenProjections(t *testing.T) {
	fast, err := tinyInt8FastEvaluator()
	if err != nil {
		t.Fatal(err)
	}
	packed, err := packQ8Layers(fast.w)
	if err != nil {
		t.Fatal(err)
	}
	packed[0].down = nil
	if compactQ8ProjectionWeights(fast.w, packed) {
		t.Fatal("compaction accepted a missing down projection")
	}
	for _, mat := range layerProjectionMats(&fast.w.Layers[0]) {
		if mat.Kind() != "int8" {
			t.Fatalf("failed compaction changed a canonical matrix to %q", mat.Kind())
		}
	}
}

func TestOfficialSMEOwnedCompactionParityAndZeroAlloc(t *testing.T) {
	fx := loadOfficialSMEFixture(t)
	lengths := []int{1, 12, 17, 128}
	idsByLength := make([][]int, 0, len(lengths))
	refs := make([][]float32, 0, len(lengths))
	for _, n := range lengths {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = fx.ids[i%len(fx.ids)]
		}
		ref, err := fx.encoder.model.HiddenLast(ids)
		if err != nil {
			t.Fatalf("goinfer reference %d-token hidden: %v", n, err)
		}
		idsByLength = append(idsByLength, ids)
		refs = append(refs, ref)
	}
	if !fx.encoder.prefill.compactOwnedWeights() {
		t.Fatal("owning Encoder could not compact its complete SME projection layout")
	}
	if !fx.encoder.prefill.weightsCompacted {
		t.Fatal("Encoder prefill did not record compacted layout")
	}
	for layer := range fx.encoder.model.Weights().Layers {
		for _, mat := range layerProjectionMats(&fx.encoder.model.Weights().Layers[layer]) {
			if mat.Kind() != "" {
				t.Fatalf("layer %d retained canonical %q projection", layer, mat.Kind())
			}
		}
	}
	runtime.GC()
	var compacted runtime.MemStats
	runtime.ReadMemStats(&compacted)
	t.Logf("canonical-drop Go heap: before=%d MiB after-GC=%d MiB packed=%d MiB pack=%s", fx.heapBytes/(1<<20), compacted.HeapAlloc/(1<<20), fx.packedByte/(1<<20), fx.packTime)

	for i, ids := range idsByLength {
		got := make([]float32, hiddenSize)
		sequences, destinations := [][]int{ids}, [][]float32{got}
		if err := fx.encoder.EmbedTokensInto(context.Background(), clm.StateRole, sequences, destinations); err != nil {
			t.Fatalf("compacted %d-token Encoder inference: %v", len(ids), err)
		}
		cos, maxAbs, rms := hiddenDiff(got, refs[i])
		t.Logf("compacted %d-token vs goinfer: cosine=%.9f max_abs=%.6g rms=%.6g", len(ids), cos, maxAbs, rms)
		if cos < 0.9999 || maxAbs > 0.3 {
			t.Fatalf("compacted %d-token output diverged from goinfer: cosine=%.9f max_abs=%g", len(ids), cos, maxAbs)
		}
		if got := testing.AllocsPerRun(5, func() {
			if err := fx.encoder.EmbedTokensInto(context.Background(), clm.StateRole, sequences, destinations); err != nil {
				panic(err)
			}
		}); got != 0 {
			t.Fatalf("compacted public %d-token inference allocated %.2f times/call", len(ids), got)
		}
	}
	if headPath := os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE"); headPath != "" {
		head, err := clm.Load(headPath)
		if err != nil {
			t.Fatalf("load official CLM head: %v", err)
		}
		engine, err := clm.NewEngine(head, fx.encoder)
		if err != nil {
			t.Fatal(err)
		}
		state := "What causes tides on Earth?"
		candidates := []string{
			"The Moon’s gravitational pull.",
			"Photosynthesis in plants.",
			"Because the Earth is round.",
		}
		ranks, err := engine.Rank(context.Background(), state, candidates, 1)
		if err != nil {
			t.Fatalf("rank with compacted encoder: %v", err)
		}
		want := []string{"The Moon’s gravitational pull.", "Because the Earth is round.", "Photosynthesis in plants."}
		for i, candidate := range want {
			if ranks[i].Candidate != candidate || ranks[i].Rank != i+1 {
				t.Fatalf("compacted CLM rank %d = %+v, want %q", i+1, ranks[i], candidate)
			}
		}
	}
}

func TestOfficialSMECompactionRSSRelease(t *testing.T) {
	fx := loadOfficialSMEFixture(t)
	before, beforeOK := processRSSKiB(t)
	if !fx.encoder.prefill.compactOwnedWeights() {
		t.Fatal("owning Encoder could not compact its complete SME projection layout")
	}
	runtime.GC()
	afterGC, afterGCOK := processRSSKiB(t)
	start := time.Now()
	debug.FreeOSMemory()
	freeOSMemoryTime := time.Since(start)
	afterFree, afterFreeOK := processRSSKiB(t)
	if beforeOK && afterGCOK && afterFreeOK {
		t.Logf("Qwen compaction RSS: before=%d MiB after-runtime.GC=%d MiB after-FreeOSMemory=%d MiB; FreeOSMemory=%s", before>>10, afterGC>>10, afterFree>>10, freeOSMemoryTime)
	} else {
		t.Logf("Qwen compaction RSS unavailable in this environment; FreeOSMemory=%s", freeOSMemoryTime)
	}
}

func processRSSKiB(tb testing.TB) (uint64, bool) {
	tb.Helper()
	output, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		tb.Logf("read test process RSS: %v", err)
		return 0, false
	}
	rss, err := strconv.ParseUint(strings.TrimSpace(string(output)), 10, 64)
	if err != nil {
		tb.Logf("parse test process RSS %q: %v", output, err)
		return 0, false
	}
	return rss, true
}

func BenchmarkOfficialSMEOwnedCompactedOneAndTwelveTokens(b *testing.B) {
	fx := loadOfficialSMEFixture(b)
	if !fx.encoder.prefill.compactOwnedWeights() {
		b.Fatal("owning Encoder could not compact its complete SME projection layout")
	}
	runtime.GC()
	var compacted runtime.MemStats
	runtime.ReadMemStats(&compacted)
	b.Logf("compacted Go heap=%d MiB, heap before drop=%d MiB, packed=%d MiB, pack=%s, pid=%d", compacted.HeapAlloc/(1<<20), fx.heapBytes/(1<<20), fx.packedByte/(1<<20), fx.packTime, os.Getpid())
	b.ReportMetric(float64(fx.packTime.Nanoseconds()), "pack-ns")
	b.ReportMetric(float64(fx.heapBytes-compacted.HeapAlloc)/(1<<20), "heap-drop-MiB")
	b.ReportAllocs()
	for _, item := range []struct {
		name string
		ids  []int
	}{
		{name: "1token", ids: fx.ids[:1]},
		{name: "12tokens", ids: fx.ids},
	} {
		ids := item.ids
		destination := make([]float32, hiddenSize)
		sequences, destinations := [][]int{ids}, [][]float32{destination}
		call := func() {
			if err := fx.encoder.EmbedTokensInto(context.Background(), clm.StateRole, sequences, destinations); err != nil {
				panic(err)
			}
		}
		call()
		if allocs := testing.AllocsPerRun(5, call); allocs != 0 {
			b.Fatalf("compacted %s inference allocated %.2f times/call", item.name, allocs)
		}
		b.Run(item.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				call()
			}
		})
	}
}

func tinyInt8FastEvaluator() (*FastEvaluator, error) {
	const hidden, vocab, intermediate = 8, 4, 12
	cfg := &decoder.Config{
		ModelType: "qwen3", VocabSize: vocab, HiddenDim: hidden, NumLayers: 1,
		NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: intermediate,
		MaxPositions: 128, RMSNormEps: 1e-6, RoPEGlobalBase: 1e6,
	}
	mat := func(rows, cols int, seed float32) linalg.WeightMat {
		data := make([]float32, rows*cols)
		for i := range data {
			data[i] = seed * float32((i%7)-3) / 10
		}
		return linalg.QuantizeInt8(data, rows, cols, false)
	}
	embedData := make([]float32, vocab*hidden)
	for i := range embedData {
		embedData[i] = float32(i%11-5) / 8
	}
	norm := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}
	weights := &decoder.Weights{
		Cfg: *cfg, Embed: linalg.WrapF32(embedData, vocab, hidden), FinalNorm: norm(hidden),
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
