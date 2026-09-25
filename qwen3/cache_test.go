// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"math/rand"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

// TestEmbeddingCacheMatchesModel drives random inserts, hits, and evictions
// against a reference map of the entries that must still be present.
func TestEmbeddingCacheMatchesModel(t *testing.T) {
	const entries, width = 37, 5
	c := newEmbeddingCache(entries, width)
	rng := rand.New(rand.NewSource(9))
	vec := func(id int) []float32 {
		v := make([]float32, width)
		for i := range v {
			v[i] = float32(id*10 + i)
		}
		return v
	}
	idsOf := func(id int) []int { return []int{id % 7, id, id / 3} }
	present := map[int]bool{}
	got := make([]float32, width)
	for range 20000 {
		id := rng.Intn(120)
		if c.get(idsOf(id), got) {
			if !present[id] {
				t.Fatalf("hit for evicted or never-inserted id %d", id)
			}
			if !slices.Equal(got, vec(id)) {
				t.Fatalf("id %d returned %v", id, got)
			}
			continue
		}
		c.put(idsOf(id), vec(id))
		present[id] = true
		// Rebuild the truth from the cache's own key set.
		live := map[cacheKey]bool{}
		for e := range c.used {
			live[c.keys[e]] = true
		}
		for p := range present {
			if !live[c.key(idsOf(p))] {
				delete(present, p)
			}
		}
		if len(present) > entries {
			t.Fatalf("%d live entries exceed capacity %d", len(present), entries)
		}
		// Every live key must be reachable through the index.
		for p := range present {
			if _, e := c.find(c.key(idsOf(p))); e < 0 {
				t.Fatalf("live id %d is unreachable after probing", p)
			}
		}
	}
	if c.hits == 0 || c.used != entries {
		t.Fatalf("hits=%d used=%d", c.hits, c.used)
	}
}

func TestEncoderCacheServesRepeatsWithoutAllocating(t *testing.T) {
	ck := lmtest.Write(t, 5)
	m, err := LoadWeights(ck.Dir, "")
	if err != nil {
		t.Fatal(err)
	}
	enc, err := newModel(m, nil, 2, 16, lmtest.Shape.MaxPos)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	ids := [][]int{{1, 2, 3}, {4, 5}, {1, 2, 3}, {6}}
	dst := make([][]float32, len(ids))
	for i := range dst {
		dst[i] = make([]float32, lmtest.Shape.Hidden)
	}
	embed := func() {
		enc.mu.Lock()
		defer enc.mu.Unlock()
		enc.batchIDs = append(enc.batchIDs[:0], ids...)
		if err := enc.embedBatchLocked(context.Background(), enc.batchIDs, dst); err != nil {
			panic(err)
		}
	}
	embed()
	want := make([]float32, lmtest.Shape.Hidden)
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(1)
	for i, seq := range ids {
		if err := e.HiddenLastInto(seq, want, ws); err != nil {
			t.Fatal(err)
		}
		// Batched and single evaluations may use different kernels for the
		// pruned last layer (tile vs one-row GEMV), so sums reassociate.
		if cos, maxAbs := lmtest.VectorParity(dst[i], want); cos < 0.9999999 || maxAbs > 2e-3 {
			t.Fatalf("input %d differs from a fresh evaluation: cosine=%.9f max_abs=%g", i, cos, maxAbs)
		}
	}
	first := slices.Clone(dst[0])
	hits0, _ := enc.CacheStats()
	embed()
	hits1, lookups := enc.CacheStats()
	if hits1-hits0 != uint64(len(ids)) {
		t.Fatalf("second call hit %d of %d inputs (lookups %d)", hits1-hits0, len(ids), lookups)
	}
	if !slices.Equal(first, dst[0]) {
		t.Fatal("cached vector differs from the computed one")
	}
	if allocs := testing.AllocsPerRun(20, embed); allocs != 0 {
		t.Fatalf("cached Embed allocated %.2f times/call", allocs)
	}
}

// TestEncoderMixesPrefixAndBatchedInputs checks that long inputs routed
// through the prefix store and short inputs batched together all land in the
// caller's destinations in order, and that a growing input reuses its prefix.
func TestEncoderMixesPrefixAndBatchedInputs(t *testing.T) {
	ck := lmtest.Write(t, 7)
	m, err := LoadWeights(ck.Dir, "")
	if err != nil {
		t.Fatal(err)
	}
	for _, cacheEntries := range []int{-1, 16} {
		enc, err := newModel(m, nil, 3, cacheEntries, lmtest.Shape.MaxPos)
		if err != nil {
			t.Fatal(err)
		}
		conv := make([]int, 200)
		for i := range conv {
			conv[i] = (i*3 + i/11) % lmtest.Shape.Vocab
		}
		ids := [][]int{{1, 2}, conv[:120], {4, 5, 6}, conv[:90:90]}
		dst := make([][]float32, len(ids))
		for i := range dst {
			dst[i] = make([]float32, lmtest.Shape.Hidden)
		}
		heads := slices.Clone(dst) // the caller's slice headers must not move
		embed := func(in [][]int) {
			enc.mu.Lock()
			defer enc.mu.Unlock()
			enc.batchIDs = append(enc.batchIDs[:0], in...)
			if err := enc.embedBatchLocked(context.Background(), enc.batchIDs, dst[:len(in)]); err != nil {
				panic(err)
			}
		}
		embed(ids)
		e, _ := NewEvaluator(m)
		ws, _ := e.NewWorkspace(1)
		want := make([]float32, lmtest.Shape.Hidden)
		for i, seq := range ids {
			if &dst[i][0] != &heads[i][0] {
				t.Fatalf("destination %d was reordered", i)
			}
			if err := e.HiddenLastInto(seq, want, ws); err != nil {
				t.Fatal(err)
			}
			if cos, maxAbs := lmtest.VectorParity(dst[i], want); cos < 0.9999999 || maxAbs > 2e-3 {
				t.Fatalf("cache=%d input %d: cosine=%.9f max_abs=%g", cacheEntries, i, cos, maxAbs)
			}
		}
		// The stored prefix now holds conv[:90]; growing it reuses 90 tokens.
		r0, c0 := enc.PrefixStats()
		embed([][]int{conv[:200]})
		r1, c1 := enc.PrefixStats()
		if r1-r0 != 90 || c1-c0 != 110 {
			t.Fatalf("cache=%d growth reused %d and computed %d tokens, want 90 and 110", cacheEntries, r1-r0, c1-c0)
		}
		if err := e.HiddenLastInto(conv[:200], want, ws); err != nil {
			t.Fatal(err)
		}
		if cos, maxAbs := lmtest.VectorParity(dst[0], want); cos < 0.9999999 || maxAbs > 2e-3 {
			t.Fatalf("grown conversation: cosine=%.9f max_abs=%g", cos, maxAbs)
		}
		if allocs := testing.AllocsPerRun(5, func() { embed(ids) }); allocs != 0 {
			t.Fatalf("cache=%d: warmed mixed Embed allocated %.2f times", cacheEntries, allocs)
		}
		_ = ws.Close()
		_ = enc.Close()
	}
}
