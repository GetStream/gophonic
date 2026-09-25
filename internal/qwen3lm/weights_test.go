// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

func TestLoadModelExactF16Weights(t *testing.T) {
	ck := lmtest.Write(t, 1)
	m, err := LoadWeights(ck.Dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if m.Format() != WeightsF16 {
		t.Fatalf("default format %q", m.Format())
	}
	// Every BF16 projection weight must round-trip exactly through packing.
	want := ck.Tensors["model.layers.1.mlp.down_proj.weight"]
	w := m.layers[1].down.f16
	k, n := w.Dims()
	x := make([]float32, k)
	got := make([]float32, n)
	ws, _ := q8gemm.NewWorkspace(k)
	for col := range k {
		clear(x)
		x[col] = 1
		if err := q8gemm.MulInto(got, x, 1, w, ws); err != nil {
			t.Fatal(err)
		}
		for row := range n {
			if got[row] != want[row*k+col] {
				t.Fatalf("down[%d,%d] = %g, want exact %g", row, col, got[row], want[row*k+col])
			}
		}
	}
	for i, v := range ck.Tensors["model.norm.weight"] {
		if m.finalNorm[i] != v {
			t.Fatalf("final norm %d = %g, want %g", i, m.finalNorm[i], v)
		}
	}
}

func TestLoadModelRejectsBadCheckpoints(t *testing.T) {
	ck := lmtest.Write(t, 2)
	if _, err := LoadWeights(ck.Dir, "int4"); err == nil {
		t.Fatal("accepted an unknown weight format")
	}
	cfg := filepath.Join(ck.Dir, "config.json")
	raw, _ := os.ReadFile(cfg)
	for _, bad := range []string{`"rope_scaling":{"type":"yarn"}`, `"attention_bias":true`} {
		patched := []byte(string(raw[:len(raw)-1]) + "," + bad + "}")
		// Later duplicate keys are rejected or override; either way the load must fail.
		if err := os.WriteFile(cfg, patched, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadWeights(ck.Dir, ""); err == nil {
			t.Fatalf("accepted config with %s", bad)
		}
	}
	if err := os.WriteFile(cfg, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(filepath.Join(ck.Dir, "model.safetensors"), 4096); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadWeights(ck.Dir, ""); err == nil {
		t.Fatal("accepted a truncated safetensors file")
	}
}

// TestEvaluatorMatchesReference checks the full forward pass against the
// float64 oracle, for single sequences, packed batches spanning several
// 16-row tiles, every worker count, and both weight formats.
func TestEvaluatorMatchesReference(t *testing.T) {
	testEvaluatorMatchesReference(t)
}

// TestEvaluatorPortableMatchesReference repeats the oracle check on the
// kernel used by CPUs without SME.
func TestEvaluatorPortableMatchesReference(t *testing.T) {
	q8gemm.ForcePortableForTesting(true)
	defer q8gemm.ForcePortableForTesting(false)
	testEvaluatorMatchesReference(t)
}

func testEvaluatorMatchesReference(t *testing.T) {
	ck := lmtest.Write(t, 3)
	long := func(n, seed int) []int {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7 + seed) % lmtest.Shape.Vocab
		}
		return ids
	}
	// 70 and 290 tokens take the blocked GEMM attention path (three blocks,
	// the last partial); the rest use the streaming path in the same batch.
	seqs := [][]int{{1}, {2, 3, 0, 1, 2}, long(70, 3), {3, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 1, 3, 2, 22, 7, 9}, {0, 2}, long(290, 5), {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
	want := make([][]float32, len(seqs))
	for i, ids := range seqs {
		want[i] = ck.ReferenceHidden(ids)
	}
	for _, tc := range []struct {
		format        string
		cosine, abs64 float64
	}{
		// FP16 activations (11-bit significand) with exact weights.
		{WeightsF16, 0.99999, 2e-2},
		// Per-row int8 weights on random Gaussian rows.
		{WeightsInt8, 0.999, 1e-1},
	} {
		m, err := LoadWeights(ck.Dir, tc.format)
		if err != nil {
			t.Fatal(err)
		}
		e, _ := NewEvaluator(m)
		for _, workers := range []int{1, 3, 8} {
			ws, err := e.NewWorkspace(workers)
			if err != nil {
				t.Fatal(err)
			}
			got := make([][]float32, len(seqs))
			for i := range got {
				got[i] = make([]float32, lmtest.Shape.Hidden)
			}
			if err := e.HiddenLastBatchInto(seqs, got, ws); err != nil {
				t.Fatal(err)
			}
			single := make([]float32, lmtest.Shape.Hidden)
			for i := range seqs {
				cos, maxAbs := lmtest.VectorParity(got[i], want[i])
				if cos < tc.cosine || maxAbs > tc.abs64 {
					t.Fatalf("%s workers=%d seq %d: cosine=%.9f max_abs=%g", tc.format, workers, i, cos, maxAbs)
				}
				if err := e.HiddenLastInto(seqs[i], single, ws); err != nil {
					t.Fatal(err)
				}
				if cos, maxAbs := lmtest.VectorParity(single, got[i]); cos < 0.9999999 || maxAbs > 3e-4 {
					t.Fatalf("%s workers=%d seq %d: separate vs batched cosine=%.9f max_abs=%g", tc.format, workers, i, cos, maxAbs)
				}
			}
			if allocs := testing.AllocsPerRun(10, func() {
				if err := e.HiddenLastBatchInto(seqs, got, ws); err != nil {
					panic(err)
				}
			}); allocs != 0 {
				t.Fatalf("%s workers=%d: warmed batch allocated %.2f times/call", tc.format, workers, allocs)
			}
			if err := ws.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestEvaluatorRejectsInvalidInput(t *testing.T) {
	ck := lmtest.Write(t, 4)
	m, err := LoadWeights(ck.Dir, "")
	if err != nil {
		t.Fatal(err)
	}
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(2)
	defer ws.Close()
	dst := make([]float32, lmtest.Shape.Hidden)
	for name, ids := range map[string][]int{
		"empty": {}, "negative": {1, -1}, "vocab": {lmtest.Shape.Vocab}, "context": make([]int, lmtest.Shape.MaxPos+1),
	} {
		if err := e.HiddenLastInto(ids, dst, ws); err == nil {
			t.Errorf("%s: accepted invalid tokens", name)
		}
	}
	if err := e.HiddenLastInto([]int{1}, dst[:3], ws); err == nil {
		t.Error("accepted a short destination")
	}
	other, _ := NewEvaluator(m)
	if err := other.HiddenLastInto([]int{1}, dst, ws); err == nil {
		t.Error("accepted another evaluator's workspace")
	}
}

// TestPrefixExtensionMatchesFullEvaluation grows, branches, and restarts a
// stored prefix and checks every result against a fresh full evaluation.
func TestPrefixExtensionMatchesFullEvaluation(t *testing.T) {
	ck := lmtest.Write(t, 6)
	for _, format := range []string{WeightsF16, WeightsInt8} {
		m, err := LoadWeights(ck.Dir, format)
		if err != nil {
			t.Fatal(err)
		}
		e, _ := NewEvaluator(m)
		ws, _ := e.NewWorkspace(3)
		kv, err := e.NewPrefixKV(lmtest.Shape.MaxPos)
		if err != nil {
			t.Fatal(err)
		}
		seq := func(n, seed int) []int {
			ids := make([]int, n)
			for i := range ids {
				ids[i] = (i*5 + seed + i/7) % lmtest.Shape.Vocab
			}
			return ids
		}
		a := seq(230, 1)
		b := append(append([]int(nil), a[:90]...), seq(40, 9)...) // branches after 90 tokens
		got, fresh := make([]float32, lmtest.Shape.Hidden), make([]float32, lmtest.Shape.Hidden)
		check := func(name string, full []int) {
			t.Helper()
			if err := e.HiddenLastInto(full, fresh, ws); err != nil {
				t.Fatal(err)
			}
			// Extensions always use blocked attention while short fresh
			// sequences stream, so summation order (and an occasional FP16
			// activation rounding) differs.
			if cos, maxAbs := lmtest.VectorParity(got, fresh); cos < 0.9999999 || maxAbs > 2e-3 {
				t.Fatalf("%s %s: extension vs fresh cosine=%.9f max_abs=%g", format, name, cos, maxAbs)
			}
			refGate := 0.99999
			if format == WeightsInt8 {
				refGate = 0.999 // per-row int8 weights are lossy by design
			}
			if cos, _ := lmtest.VectorParity(got, ck.ReferenceHidden(full)); cos < refGate {
				t.Fatalf("%s %s: extension vs float64 reference cosine=%.9f", format, name, cos)
			}
		}
		steps := []struct {
			name string
			keep int
			full []int
		}{
			{"cold 150", 0, a[:150]},
			{"append 80", 150, a},
			{"branch at 90", 90, b},
			{"one token", len(b), append(b[:len(b):len(b)], 3)},
			{"restart", 0, a[:20]},
		}
		for _, st := range steps {
			if n := kv.CommonPrefix(st.full); n < st.keep {
				t.Fatalf("%s: stored prefix shares %d tokens, want at least %d", st.name, n, st.keep)
			}
			if err := e.HiddenLastExtendInto(kv, st.keep, st.full[st.keep:], got, ws); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(kv.Tokens(), st.full) {
				t.Fatalf("%s: stored tokens not updated", st.name)
			}
			check(st.name, st.full)
		}
		tail := a[200:]
		if err := e.HiddenLastExtendInto(kv, 0, a[:200], got, ws); err != nil {
			t.Fatal(err)
		}
		if allocs := testing.AllocsPerRun(5, func() {
			if err := e.HiddenLastExtendInto(kv, 200, tail, got, ws); err != nil {
				panic(err)
			}
		}); allocs != 0 {
			t.Fatalf("%s: warmed extension allocated %.2f times", format, allocs)
		}
		if err := e.HiddenLastExtendInto(kv, len(kv.Tokens())+1, a[:1], got, ws); err == nil {
			t.Fatal("accepted keep beyond stored tokens")
		}
		if err := e.HiddenLastExtendInto(kv, 0, seq(lmtest.Shape.MaxPos+1, 0), got, ws); err == nil {
			t.Fatal("accepted more tokens than the prefix capacity")
		}
		_ = ws.Close()
	}
}

// TestSharedPrefixMatchesFullEvaluation checks HiddenLastSharedInto against
// fresh evaluation of prefix+sequence for short and multi-block sequences,
// and that the shared prefix is left untouched.
func TestSharedPrefixMatchesFullEvaluation(t *testing.T) {
	ck := lmtest.Write(t, 8)
	for _, format := range []string{WeightsF16, WeightsInt8} {
		m, err := LoadWeights(ck.Dir, format)
		if err != nil {
			t.Fatal(err)
		}
		e, _ := NewEvaluator(m)
		ws, _ := e.NewWorkspace(3)
		mk := func(n, seed int) []int {
			ids := make([]int, n)
			for i := range ids {
				ids[i] = (i*3 + seed + i/5) % lmtest.Shape.Vocab
			}
			return ids
		}
		for _, prefixLen := range []int{5, 40, 150} {
			prefix := mk(prefixLen, 1)
			kv, err := e.NewPrefixKV(prefixLen)
			if err != nil {
				t.Fatal(err)
			}
			scratch := make([]float32, lmtest.Shape.Hidden)
			if err := e.HiddenLastExtendInto(kv, 0, prefix, scratch, ws); err != nil {
				t.Fatal(err)
			}
			before := slices.Clone(kv.keys[1][:prefixLen*m.cfg.kvDim])
			seqs := [][]int{mk(1, 2), mk(9, 3), mk(130, 4), mk(3, 5)}
			got := make([][]float32, len(seqs))
			for i := range got {
				got[i] = make([]float32, lmtest.Shape.Hidden)
			}
			if err := e.HiddenLastSharedInto(kv, seqs, got, ws); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(before, kv.keys[1][:prefixLen*m.cfg.kvDim]) || !slices.Equal(kv.Tokens(), prefix) {
				t.Fatalf("%s prefix %d: shared evaluation modified the prefix", format, prefixLen)
			}
			want := make([]float32, lmtest.Shape.Hidden)
			for i, seq := range seqs {
				full := append(slices.Clone(prefix), seq...)
				if err := e.HiddenLastInto(full, want, ws); err != nil {
					t.Fatal(err)
				}
				// int8 rounds each row's activations, which turns the two
				// attention paths' different summation orders into small
				// differences; exact weights must agree tightly.
				minCos, maxDiff := 0.9999999, 2e-3
				if format == WeightsInt8 {
					minCos, maxDiff = 0.9999, 0.05
				}
				if cos, maxAbs := lmtest.VectorParity(got[i], want); cos < minCos || maxAbs > maxDiff {
					t.Fatalf("%s prefix %d seq %d: shared vs fresh cosine=%.9f max_abs=%g", format, prefixLen, i, cos, maxAbs)
				}
			}
			if allocs := testing.AllocsPerRun(5, func() {
				if err := e.HiddenLastSharedInto(kv, seqs, got, ws); err != nil {
					panic(err)
				}
			}); allocs != 0 {
				t.Fatalf("%s: shared evaluation allocated %.2f times", format, allocs)
			}
		}
		_ = ws.Close()
	}
}
