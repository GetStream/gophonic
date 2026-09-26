// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

// A greedy run is the model's own greedy continuation, every step in one
// submission: each step's logits are the reference's for the tokens so
// far, and each token is its step's likeliest.
func TestDecodeMatchesReference(t *testing.T) {
	ck := lmtest.WriteShape(t, 7, lmtest.GPUShape, nil)
	s := ck.Geometry
	if _, err := Load(ck.Dir, LoadOptions{Format: WeightsF16}); err != nil {
		t.Fatal(err)
	}
	cpu, _ := Load(ck.Dir, LoadOptions{Format: WeightsF16})
	ce, _ := NewEvaluator(cpu)
	if _, err := ce.NewDecoder([][]float32{ck.Tensors["lm_head.weight"]}, nil, s.Vocab); err == nil {
		t.Fatal("a decoder for weights on the CPU")
	}
	cpu.Release()
	head, embed := ck.Tensors["lm_head.weight"], ck.Tensors["model.embed_tokens.weight"]
	ids := []int{3, 14, 15, 92, 65, 35}
	for _, format := range []string{WeightsGPU, WeightsGPUQ8} {
		m, err := Load(ck.Dir, LoadOptions{Format: format})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := NewEvaluator(m)
		ws, err := e.NewWorkspace(2)
		if err != nil {
			t.Fatal(err)
		}
		// A draft model: its head and embedding table at every step.
		d, err := e.NewDecoder([][]float32{head, head, head, head}, [][]float32{embed, embed, embed}, s.Vocab)
		if err != nil {
			t.Fatal(err)
		}
		kv, _ := e.NewPrefixKV(64)
		hidden := make([]float32, s.Hidden)
		if err := e.HiddenLastExtendInto(kv, 0, ids[:2], hidden, ws); err != nil {
			t.Fatal(err)
		}
		tokens := make([]int, 4)
		logits := make([]float32, len(tokens)*s.Vocab)
		if err := e.DecodeInto(d, kv, 2, ids[2:], Embeds{}, Sampling{}, tokens, logits, ws); err != nil {
			t.Fatal(err)
		}
		wantFirst := slices.Clone(logits[:s.Vocab])
		seq := append([]int(nil), ids...)
		for i, tok := range tokens {
			got := logits[i*s.Vocab : (i+1)*s.Vocab]
			want := ck.ReferenceLogits(ck.ReferenceHidden(seq))
			if cos, _ := lmtest.VectorParity(got, want); cos < 0.999 {
				t.Fatalf("%s step %d: logits cosine %.6f to the reference", format, i, cos)
			}
			if best := argmax(got); tok != best {
				t.Fatalf("%s step %d: token %d, the likeliest is %d", format, i, tok, best)
			}
			seq = append(seq, tok)
		}
		if n := len(kv.Tokens()); n != len(ids)+len(tokens)-1 {
			t.Fatalf("%s: %d tokens stored, want %d", format, n, len(ids)+len(tokens)-1)
		}
		// Continuing the run from what it stored matches it too.
		next := make([]int, 1)
		if err := e.DecodeInto(d, kv, len(kv.Tokens()), tokens[len(tokens)-1:], Embeds{}, Sampling{}, next, logits[:s.Vocab], ws); err != nil {
			t.Fatal(err)
		}
		if cos, _ := lmtest.VectorParity(logits[:s.Vocab], ck.ReferenceLogits(ck.ReferenceHidden(seq))); cos < 0.999 {
			t.Fatalf("%s: a continued run's logits have cosine %.6f", format, cos)
		}
		// A one-step decoder must match the first step exactly without
		// allocating a vocabulary-sized table that it never reads.
		one, err := e.NewDecoder([][]float32{head}, nil, s.Vocab)
		if err != nil {
			t.Fatal(err)
		}
		if one.gpu.tables != nil || one.gpu.sumsq != nil {
			t.Fatal("one-step decoder allocated unused input tables")
		}
		if err := e.HiddenLastExtendInto(kv, 0, ids[:2], hidden, ws); err != nil {
			t.Fatal(err)
		}
		if err := e.DecodeInto(one, kv, 2, ids[2:], Embeds{}, Sampling{}, next, logits[:s.Vocab], ws); err != nil {
			t.Fatal(err)
		}
		for i, want := range wantFirst {
			if math.Float32bits(logits[i]) != math.Float32bits(want) {
				t.Fatalf("%s: table-free first-step logit %d changed", format, i)
			}
		}
		if next[0] != argmax(wantFirst) {
			t.Fatalf("%s: table-free token changed", format)
		}
		one.Close()
		d.Close()
		kv.Close()
		ws.Close()
		m.Release()
	}
}

// Draws follow the top-k distribution at the temperature, and equal inputs
// draw equal tokens.
func TestDecodeSamples(t *testing.T) {
	ck := lmtest.WriteShape(t, 8, lmtest.GPUShape, nil)
	s := ck.Geometry
	m, err := Load(ck.Dir, LoadOptions{Format: WeightsGPUQ8})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Release()
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(2)
	defer ws.Close()
	// A head scaled so that the top few share the probability.
	head := make([]float32, len(ck.Tensors["lm_head.weight"]))
	for i, v := range ck.Tensors["lm_head.weight"] {
		head[i] = v * 3
	}
	d, err := e.NewDecoder([][]float32{head}, nil, s.Vocab)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	kv, _ := e.NewPrefixKV(16)
	defer kv.Close()
	ids := []int{5, 9, 2}
	logits := make([]float32, s.Vocab)
	tok := make([]int, 1)
	const k, temp, runs = 6, 0.7, 3000
	counts := make([]int, s.Vocab)
	for r := range runs {
		if err := e.DecodeInto(d, kv, 0, ids, Embeds{}, Sampling{TopK: k, Temperature: temp, Seed: 42, Draw: uint64(r)}, tok, logits, ws); err != nil {
			t.Fatal(err)
		}
		counts[tok[0]]++
	}
	// The expected distribution: softmax(logits / temp) over the top k.
	order := make([]int, s.Vocab)
	for i := range order {
		order[i] = i
	}
	for i := range k {
		for j := i + 1; j < len(order); j++ {
			if logits[order[j]] > logits[order[i]] {
				order[i], order[j] = order[j], order[i]
			}
		}
	}
	var z float64
	p := make([]float64, k)
	for i := range k {
		p[i] = math.Exp(float64(logits[order[i]]-logits[order[0]]) / temp)
		z += p[i]
	}
	inTop := 0
	for i := range k {
		got, want := float64(counts[order[i]])/runs, p[i]/z
		inTop += counts[order[i]]
		t.Logf("token %d: drawn %.3f, expected %.3f", order[i], got, want)
		if math.Abs(got-want) > 0.035 {
			t.Errorf("token %d drawn %.3f of the time; want %.3f", order[i], got, want)
		}
	}
	if inTop != runs {
		t.Fatalf("%d of %d draws outside the top %d", runs-inTop, runs, k)
	}
	// The same draw again.
	a, b := make([]int, 1), make([]int, 1)
	e.DecodeInto(d, kv, 0, ids, Embeds{}, Sampling{TopK: k, Temperature: temp, Seed: 7, Draw: 11}, a, nil, ws)
	e.DecodeInto(d, kv, 0, ids, Embeds{}, Sampling{TopK: k, Temperature: temp, Seed: 7, Draw: 11}, b, nil, ws)
	if a[0] != b[0] {
		t.Fatalf("equal draws gave %d and %d", a[0], b[0])
	}
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// Lanes share a Decoder: each run's tokens are its workspace's.
func TestDecodeConcurrentLanes(t *testing.T) {
	ck := lmtest.WriteShape(t, 9, lmtest.GPUShape, nil)
	s := ck.Geometry
	m, err := Load(ck.Dir, LoadOptions{Format: WeightsGPUQ8})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Release()
	e, _ := NewEvaluator(m)
	head, embed := ck.Tensors["lm_head.weight"], ck.Tensors["model.embed_tokens.weight"]
	d, err := e.NewDecoder([][]float32{head, head, head}, [][]float32{embed, embed}, s.Vocab)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	run := func(ids []int, ws *Workspace, kv *PrefixKV, dst []int) error {
		return e.DecodeInto(d, kv, 0, ids, Embeds{}, Sampling{TopK: 4, Temperature: 0.8, Seed: 3}, dst, nil, ws)
	}
	inputs := [][]int{{1, 2, 3}, {7, 8, 9, 10}}
	want := make([][]int, len(inputs))
	errs := make(chan error, len(inputs))
	for i, ids := range inputs {
		ws, _ := e.NewWorkspace(1)
		defer ws.Close()
		kv, _ := e.NewPrefixKV(16)
		defer kv.Close()
		want[i] = make([]int, 3)
		if err := run(ids, ws, kv, want[i]); err != nil {
			t.Fatal(err)
		}
		go func() {
			got := make([]int, 3)
			for range 50 {
				if err := run(ids, ws, kv, got); err != nil {
					errs <- err
					return
				}
				if !slices.Equal(got, want[i]) {
					errs <- fmt.Errorf("lane %d drew %v, alone %v", i, got, want[i])
					return
				}
			}
			errs <- nil
		}()
	}
	for range inputs {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
}
