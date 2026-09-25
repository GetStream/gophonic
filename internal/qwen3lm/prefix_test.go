// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

func TestPackedPrefixBranchesAndReusedCopies(t *testing.T) {
	ck := lmtest.Write(t, 29)
	for _, format := range []string{WeightsF16, WeightsInt8} {
		m, err := LoadWeights(ck.Dir, format)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Release()
		e, _ := NewEvaluator(m)
		ws, _ := e.NewWorkspace(3)
		defer ws.Close()
		seed := make([]int, 300)
		for i := range seed {
			seed[i] = (i*7 + i/9) % m.cfg.vocab
		}
		src, _ := e.NewPrefixKV(320)
		defer src.Close()
		dst, _ := e.NewPrefixKV(320)
		defer dst.Close()
		got, want := make([]float32, m.cfg.hidden), make([]float32, m.cfg.hidden)
		if err := e.HiddenLastExtendInto(src, 0, seed, got, ws); err != nil {
			t.Fatal(err)
		}
		for _, keep := range []int{0, 1, 15, 16, 17, 255, 256, 257, 299, 3} {
			// Copy into a destination that already contains different KV values.
			if err := e.HiddenLastExtendInto(dst, 0, []int{4, 5, 6, 7}, got, ws); err != nil {
				t.Fatal(err)
			}
			dst.CopyPrefix(src, keep)
			runtime.GC()
			tail := []int{1, 2}
			if err := e.HiddenLastExtendInto(dst, keep, tail, got, ws); err != nil {
				t.Fatal(err)
			}
			full := append(slices.Clone(seed[:keep]), tail...)
			if err := e.HiddenLastInto(full, want, ws); err != nil {
				t.Fatal(err)
			}
			if cos, abs := lmtest.VectorParity(got, want); cos < 0.9999999 || abs > 2e-3 {
				t.Fatalf("%s keep=%d cosine=%g abs=%g", format, keep, cos, abs)
			}
			// Branch inside that cache, including across the 256-token boundary.
			if err := e.HiddenLastExtendInto(dst, keep, []int{3, 4}, got, ws); err != nil {
				t.Fatal(err)
			}
			full = append(slices.Clone(seed[:keep]), 3, 4)
			if err := e.HiddenLastInto(full, want, ws); err != nil {
				t.Fatal(err)
			}
			if cos, abs := lmtest.VectorParity(got, want); cos < 0.9999999 || abs > 2e-3 {
				t.Fatalf("branched %s keep=%d cosine=%g abs=%g", format, keep, cos, abs)
			}
		}
	}
}

func TestPreparedPrefixHasConcurrentReadOnlyConsumers(t *testing.T) {
	ck := lmtest.Write(t, 31)
	m, err := LoadWeights(ck.Dir, WeightsF16)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Release()
	e, _ := NewEvaluator(m)
	prep, _ := e.NewWorkspace(2)
	defer prep.Close()
	kv, _ := e.NewPrefixKV(320)
	defer kv.Close()
	prefix := make([]int, 257)
	for i := range prefix {
		prefix[i] = i % m.cfg.vocab
	}
	hidden := make([]float32, m.cfg.hidden)
	if err := e.HiddenLastExtendInto(kv, 0, prefix, hidden, prep); err != nil {
		t.Fatal(err)
	}
	before := packedPrefixKeys(t, kv, 1)
	seqs := [][]int{{1}, {2, 3, 4}}
	var wg sync.WaitGroup
	for range 4 {
		ws, _ := e.NewWorkspace(2)
		wg.Go(func() {
			defer ws.Close()
			got := [][]float32{make([]float32, m.cfg.hidden), make([]float32, m.cfg.hidden)}
			var first [][]float32
			for i := 0; i < 4; i++ {
				if err := e.HiddenLastSharedInto(kv, seqs, got, ws); err != nil {
					t.Error(err)
					return
				}
				if i == 0 {
					first = [][]float32{slices.Clone(got[0]), slices.Clone(got[1])}
				} else if !slices.Equal(first[0], got[0]) || !slices.Equal(first[1], got[1]) {
					t.Error("shared reads changed outputs")
				}
			}
		})
	}
	runtime.GC()
	wg.Wait()
	if !slices.Equal(before, packedPrefixKeys(t, kv, 1)) || !slices.Equal(prefix, kv.Tokens()) {
		t.Fatal("shared readers mutated prefix")
	}
}
