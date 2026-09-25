// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"runtime"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

func TestArenaGrowthPrefixCopyAndClose(t *testing.T) {
	ck := lmtest.WriteNamed(t, 17, func(s string) string { return s })
	m, err := Load(ck.Dir, LoadOptions{Format: WeightsF16})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Release()
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(16)
	defer ws.Close()
	kv, _ := e.NewPrefixKV(64)
	defer kv.Close()
	other, _ := e.NewPrefixKV(64)
	defer other.Close()
	ids := []int{1, 2, 3}
	hidden := make([]float32, m.cfg.hidden)
	if err := e.HiddenLastExtendInto(kv, 0, ids, hidden, ws); err != nil {
		t.Fatal(err)
	}
	other.CopyPrefix(kv, len(ids))
	runtime.GC()
	want := slices.Clone(hidden)
	// Grow the workspace, unmapping the old activation arena. Stored KV is
	// independent and still supports extension after several collections.
	if err := ws.Reserve(33, 64); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		runtime.GC()
	}
	if err := e.HiddenLastExtendInto(other, 2, ids[2:], hidden, ws); err != nil {
		t.Fatal(err)
	}
	if cos, _ := lmtest.VectorParity(hidden, want); cos < 0.999999 {
		t.Fatalf("prefix changed after growth: cosine %g", cos)
	}
	if len(ws.attnScratch) != maxSMEWorkers {
		t.Fatal("scratch allocated for inactive workers")
	}
	if kv.memory.Bytes() == 0 || ws.memory.Bytes() == 0 {
		t.Fatal("missing arena")
	}
	if err := ws.Close(); err != nil {
		t.Fatal(err)
	}
	if err := other.Close(); err != nil {
		t.Fatal(err)
	}
	if err := kv.Close(); err != nil {
		t.Fatal(err)
	}
	if ws.memory != nil || kv.Capacity() != 0 {
		t.Fatal("Close retained storage")
	}
}

// Recover keys with identity multiplication so the test observes every cached
// value without depending on the packed layout or its private backing slices.
func packedPrefixKeys(t *testing.T, kv *PrefixKV, layer int) []float32 {
	t.Helper()
	hd := kv.owner.m.cfg.headDim
	n := len(kv.tokens)
	identity := make([]float32, hd*hd)
	for i := range hd {
		identity[i*hd+i] = 1
	}
	out := make([]float32, len(kv.packs[layer].keysT)*hd*n)
	for g, k := range kv.packs[layer].keysT {
		if err := k.Mul(out[g*hd*n:], n, identity, hd, hd); err != nil {
			t.Fatal(err)
		}
	}
	runtime.KeepAlive(kv)
	return out
}
