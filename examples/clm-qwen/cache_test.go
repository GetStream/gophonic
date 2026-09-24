// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"context"
	"math/rand"
	"slices"
	"testing"

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
	ck := writeTinyCheckpoint(t, 5)
	m, err := LoadModel(ck.dir, "")
	if err != nil {
		t.Fatal(err)
	}
	enc, err := newEncoder(m, nil, 2, 16)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	ids := [][]int{{1, 2, 3}, {4, 5}, {1, 2, 3}, {6}}
	dst := make([][]float32, len(ids))
	for i := range dst {
		dst[i] = make([]float32, tinyShape.hidden)
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
	want := make([]float32, tinyShape.hidden)
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(1)
	for i, seq := range ids {
		if err := e.HiddenLastInto(seq, want, ws); err != nil {
			t.Fatal(err)
		}
		if cos, maxAbs := vectorParity(dst[i], want); cos < 0.9999999 || maxAbs > 1e-4 {
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
