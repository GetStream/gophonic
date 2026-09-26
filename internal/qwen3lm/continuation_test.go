// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

// A cached audio prefix must stay in the original own-row attention product:
// making it an ordinary kept prefix changes the FP32 reduction grouping.
func TestContinuationMatchesRecomputedRowsExactly(t *testing.T) {
	ck := lmtest.Write(t, 71)
	for _, format := range []string{WeightsF16, WeightsInt8} {
		m, err := Load(ck.Dir, LoadOptions{Format: format, Head: "lm_head.weight"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Release()
		m.cfg.maxPositions = 1024
		e, _ := NewEvaluator(m)
		for _, workers := range []int{1, 3, 8} {
			for _, anchor := range []int{0, 1, 17, 255, 256, 257} {
				t.Run(fmt.Sprintf("%s/workers%d/anchor%d", format, workers, anchor), func(t *testing.T) {
					testExactContinuation(t, e, workers, anchor)
				})
			}
		}
	}
}

func testExactContinuation(t *testing.T, e *Evaluator, workers, anchor int) {
	t.Helper()
	c := &e.m.cfg
	base, _ := e.NewWorkspace(workers)
	defer base.Close()
	fast, _ := e.NewWorkspace(workers)
	defer fast.Close()
	bkv, _ := e.NewPrefixKV(1024)
	defer bkv.Close()
	fkv, _ := e.NewPrefixKV(1024)
	defer fkv.Close()
	const placeholder = 5
	prefix := make([]int, anchor)
	for i := range prefix {
		prefix[i] = (i*7 + 3) % c.vocab
	}
	one := make([]float32, c.hidden)
	if anchor > 0 {
		if err := e.HiddenLastExtendInto(bkv, 0, prefix, one, base); err != nil {
			t.Fatal(err)
		}
		fkv.CopyPrefix(bkv, anchor)
	}
	embedding := make([]float32, 416*c.hidden)
	for i := range embedding {
		embedding[i] = float32(math.Sin(float64(i)*0.13)) * float32(i%7+1)
	}
	previous := 0
	for step, audio := range []int{127, 128, 129, 208, 255, 256, 257, 384, 416} {
		k := []int{1, 3, 16, 32}[step%4]
		ids := make([]int, audio+k)
		for i := range audio {
			ids[i] = placeholder
		}
		for i := audio; i < len(ids); i++ {
			ids[i] = (i+step)%4 + 1
		}
		embeds := Embeds{Token: placeholder, Rows: embedding[:audio*c.hidden]}
		want, got := make([]float32, k*c.hidden), make([]float32, k*c.hidden)
		if err := e.HiddenTailExtendEmbedInto(bkv, anchor, ids, embeds, want, base); err != nil {
			t.Fatal(err)
		}
		if err := e.HiddenTailContinueEmbedInto(fkv, anchor, previous, ids, embeds, got, fast); err != nil {
			t.Fatal(err)
		}
		assertContinuationBits(t, "hidden", got, want)
		logits, refLogits := make([]float32, k*c.vocab), make([]float32, k*c.vocab)
		if err := e.LogitsRowsInto(got, logits, fast); err != nil {
			t.Fatal(err)
		}
		if err := e.LogitsRowsInto(want, refLogits, base); err != nil {
			t.Fatal(err)
		}
		assertContinuationBits(t, "logits", logits, refLogits)
		assertContinuationKV(t, fkv, bkv)
		previous = audio
		if step == 8 {
			if n := testing.AllocsPerRun(1, func() {
				if err := e.HiddenTailContinueEmbedInto(fkv, anchor, audio, ids, embeds, got, fast); err != nil {
					panic(err)
				}
			}); n != 0 {
				t.Fatalf("warm continuation allocations=%g", n)
			}
			assertContinuationBits(t, "warm hidden", got, want)
		}
		// Ordinary one-token decoding after the compacted tail must reset
		// all row-skipping state and keep the reconstructed cache intact.
		next := []int{(step + 9) % c.vocab}
		after, refAfter := make([]float32, c.hidden), make([]float32, c.hidden)
		if err := e.HiddenLastExtendInto(fkv, len(fkv.Tokens()), next, after, fast); err != nil {
			t.Fatal(err)
		}
		if err := e.HiddenLastExtendInto(bkv, len(bkv.Tokens()), next, refAfter, base); err != nil {
			t.Fatal(err)
		}
		assertContinuationBits(t, "ordinary decode", after, refAfter)
		assertContinuationKV(t, fkv, bkv)
	}
}

func assertContinuationBits(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length %d != %d", name, len(got), len(want))
	}
	for i, v := range want {
		if math.Float32bits(got[i]) != math.Float32bits(v) {
			t.Fatalf("%s value %d: %08x != %08x", name, i, math.Float32bits(got[i]), math.Float32bits(v))
		}
	}
}

func assertContinuationKV(t *testing.T, got, want *PrefixKV) {
	t.Helper()
	if !slices.Equal(got.Tokens(), want.Tokens()) {
		t.Fatal("cached tokens differ")
	}
	n, hd := len(got.tokens), got.owner.m.cfg.headDim
	a, b := make([]float32, n*hd), make([]float32, n*hd)
	for l := range got.packs {
		for g := range got.packs[l].keysT {
			if err := got.packs[l].keysT[g].UnpackColumns(a, hd, 0, n); err != nil {
				t.Fatal(err)
			}
			if err := want.packs[l].keysT[g].UnpackColumns(b, hd, 0, n); err != nil {
				t.Fatal(err)
			}
			assertContinuationBits(t, fmt.Sprintf("layer%d/group%d keys", l, g), a, b)
			for ch := 0; ch*prefixChunk < n; ch++ {
				rows := min(prefixChunk, n-ch*prefixChunk)
				if err := got.packs[l].values[g][ch].UnpackRows(a, hd, 0, rows); err != nil {
					t.Fatal(err)
				}
				if err := want.packs[l].values[g][ch].UnpackRows(b, hd, 0, rows); err != nil {
					t.Fatal(err)
				}
				assertContinuationBits(t, fmt.Sprintf("layer%d/group%d/chunk%d values", l, g, ch), a[:rows*hd], b[:rows*hd])
			}
		}
	}
}

func TestContinuationRejectsInvalidReuse(t *testing.T) {
	ck := lmtest.Write(t, 72)
	m, err := LoadWeights(ck.Dir, WeightsF16)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Release()
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(1)
	defer ws.Close()
	kv, _ := e.NewPrefixKV(320)
	defer kv.Close()
	ids := make([]int, 130)
	hidden := make([]float32, m.cfg.hidden)
	if err := e.HiddenLastExtendInto(kv, 0, ids, hidden, ws); err != nil {
		t.Fatal(err)
	}
	before := slices.Clone(kv.Tokens())
	for _, reuse := range []int{-1, 131} {
		if err := e.HiddenLastContinueEmbedInto(kv, 0, reuse, ids, Embeds{}, hidden, ws); err == nil {
			t.Fatalf("accepted reuse=%d", reuse)
		}
		if !slices.Equal(kv.Tokens(), before) || ws.reuse != 0 || ws.op.firstRow != 0 {
			t.Fatal("validation failure changed continuation state")
		}
	}
	ids[len(ids)-1] = -1
	if err := e.HiddenLastContinueEmbedInto(kv, 0, 128, ids, Embeds{}, hidden, ws); err == nil {
		t.Fatal("accepted an invalid suffix token")
	}
	if ws.reuse != 0 || ws.op.firstRow != 0 || ws.prefix != nil {
		t.Fatal("failed forward pass retained continuation state")
	}
	ids[len(ids)-1] = 0
	if err := e.HiddenLastExtendInto(kv, 0, ids, hidden, ws); err != nil {
		t.Fatal(err)
	}
}

func TestContinuationKeepsWarmedAttentionDispatch(t *testing.T) {
	ck := lmtest.Write(t, 73)
	m, err := LoadWeights(ck.Dir, WeightsF16)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Release()
	m.cfg.maxPositions = 1024
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(3)
	defer ws.Close()
	ref, _ := e.NewWorkspace(3)
	defer ref.Close()
	kv, _ := e.NewPrefixKV(1024)
	defer kv.Close()
	rkv, _ := e.NewPrefixKV(1024)
	defer rkv.Close()
	ids := make([]int, 899)
	for i := range ids {
		ids[i] = (i*7 + 3) % m.cfg.vocab
	}
	got, want := make([]float32, m.cfg.hidden), make([]float32, m.cfg.hidden)
	if err := e.HiddenLastExtendInto(kv, 0, ids[:897], got, ws); err != nil {
		t.Fatal(err)
	}
	if ws.attnPerHead {
		t.Fatal("test did not warm the group dispatch")
	}
	capacity := cap(ws.attnItems)
	if err := e.HiddenLastContinueEmbedInto(kv, 0, 384, ids, Embeds{}, got, ws); err != nil {
		t.Fatal(err)
	}
	if ws.attnPerHead || cap(ws.attnItems) != capacity {
		t.Fatal("reuse changed dispatch policy or grew warmed descriptors")
	}
	if err := e.HiddenLastExtendInto(rkv, 0, ids, want, ref); err != nil {
		t.Fatal(err)
	}
	assertContinuationBits(t, "dispatch threshold", got, want)
	assertContinuationKV(t, kv, rkv)
	if n := testing.AllocsPerRun(1, func() {
		if err := e.HiddenLastContinueEmbedInto(kv, 0, 384, ids, Embeds{}, got, ws); err != nil {
			panic(err)
		}
	}); n != 0 {
		t.Fatalf("warmed threshold continuation allocations=%g", n)
	}
}
