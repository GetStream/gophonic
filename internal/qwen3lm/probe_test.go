// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

// A probe reads the attention probabilities the forward pass weighs keys
// with, and leaves the pass's result unchanged.
func TestProbeMatchesReference(t *testing.T) {
	ck := lmtest.Write(t, 41)
	s := lmtest.Shape
	ids := []int{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7}
	n := len(ids)
	for _, tc := range []struct {
		format string
		tol    float64
	}{{WeightsF16, 1e-3}, {WeightsInt8, 2e-2}} {
		m, err := Load(ck.Dir, LoadOptions{Format: tc.format})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := NewEvaluator(m)
		ws, _ := e.NewWorkspace(3)
		kv, err := e.NewPrefixKV(64)
		if err != nil {
			t.Fatal(err)
		}
		hidden := make([]float32, s.Hidden)
		for _, lh := range [][2]int{{0, 0}, {1, 5}, {1, 7}} {
			if err := e.HiddenLastExtendInto(kv, 0, ids[:n-1], hidden, ws); err != nil {
				t.Fatal(err)
			}
			// Every position, then a window running past the token.
			all := Probe{Layer: lh[0], Head: lh[1], Probs: make([]float32, n)}
			if err := e.HiddenLastExtendProbeInto(kv, n-1, ids[n-1:], Embeds{}, hidden, &all, ws); err != nil {
				t.Fatal(err)
			}
			if cos, _ := lmtest.VectorParity(hidden, ck.ReferenceHidden(ids)); cos < 0.999 {
				t.Fatalf("%s: the probed pass's state has cosine %.6f to the reference", tc.format, cos)
			}
			want := ck.ReferenceAttention(ids, lh[0], lh[1])
			var sum float64
			for j, p := range all.Probs {
				sum += float64(p)
				if d := math.Abs(float64(p) - want[j]); d > tc.tol {
					t.Fatalf("%s layer %d head %d: position %d weighs %.5f, want %.5f", tc.format, lh[0], lh[1], j, p, want[j])
				}
			}
			if math.Abs(sum-1) > 1e-4 {
				t.Fatalf("%s: probabilities sum to %.6f", tc.format, sum)
			}
			window := Probe{Layer: lh[0], Head: lh[1], From: n - 3, Probs: []float32{-1, -1, -1, -1, -1}}
			if err := e.HiddenLastExtendProbeInto(kv, n-1, ids[n-1:], Embeds{}, hidden, &window, ws); err != nil {
				t.Fatal(err)
			}
			for i, p := range window.Probs {
				if want := all.Probs[min(n-3+i, n-1)]; i >= 3 && p != 0 || i < 3 && p != want {
					t.Fatalf("%s: window position %d weighs %v, want %v", tc.format, n-3+i, p, want)
				}
			}
		}
		for _, bad := range []struct {
			ids   []int
			probe Probe
		}{
			{ids[n-2:], Probe{Probs: make([]float32, 2)}},
			{ids[n-1:], Probe{Layer: s.Layers}},
			{ids[n-1:], Probe{Head: s.Heads}},
			{ids[n-1:], Probe{Probs: make([]float32, maxProbe+1)}},
		} {
			if err := e.HiddenLastExtendProbeInto(kv, n-len(bad.ids), bad.ids, Embeds{}, hidden, &bad.probe, ws); err == nil {
				t.Fatalf("probe %+v of %d tokens: no error", bad.probe, len(bad.ids))
			}
		}
		kv.Close()
		ws.Close()
		m.Release()
	}
}
