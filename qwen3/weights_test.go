// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/thesyncim/vibejson"
)

// tinyShape is a small Qwen3 geometry that still exercises grouped-query
// attention, partial output panels, and odd tile counts.
var tinyShape = struct {
	hidden, layers, heads, kvHeads, headDim, inter, vocab, maxPos int
}{hidden: 96, layers: 2, heads: 4, kvHeads: 2, headDim: 24, inter: 136, vocab: 23, maxPos: 320}

type tinyCheckpoint struct {
	dir     string
	tensors map[string][]float32 // BF16-rounded values
	shapes  map[string][]int
}

// writeTinyCheckpoint writes a random Qwen3 snapshot (config and one
// safetensors file) whose values are exactly representable in BF16.
func writeTinyCheckpoint(t testing.TB, seed int64) *tinyCheckpoint {
	t.Helper()
	s := tinyShape
	rng := rand.New(rand.NewSource(seed))
	ck := &tinyCheckpoint{dir: t.TempDir(), tensors: map[string][]float32{}, shapes: map[string][]int{}}
	add := func(name string, scale float64, offset float64, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = bf16Round(float32(offset + rng.NormFloat64()*scale))
		}
		ck.tensors[name], ck.shapes[name] = v, shape
	}
	add("model.embed_tokens.weight", 1, 0, s.vocab, s.hidden)
	add("model.norm.weight", 0.1, 1, s.hidden)
	add("lm_head.weight", 0.02, 0, s.vocab, s.hidden)
	for l := range s.layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		add(p+"input_layernorm.weight", 0.1, 1, s.hidden)
		add(p+"post_attention_layernorm.weight", 0.1, 1, s.hidden)
		add(p+"self_attn.q_norm.weight", 0.1, 1, s.headDim)
		add(p+"self_attn.k_norm.weight", 0.1, 1, s.headDim)
		add(p+"self_attn.q_proj.weight", 0.1, 0, s.heads*s.headDim, s.hidden)
		add(p+"self_attn.k_proj.weight", 0.1, 0, s.kvHeads*s.headDim, s.hidden)
		add(p+"self_attn.v_proj.weight", 0.1, 0, s.kvHeads*s.headDim, s.hidden)
		add(p+"self_attn.o_proj.weight", 0.1, 0, s.hidden, s.heads*s.headDim)
		add(p+"mlp.gate_proj.weight", 0.1, 0, s.inter, s.hidden)
		add(p+"mlp.up_proj.weight", 0.1, 0, s.inter, s.hidden)
		add(p+"mlp.down_proj.weight", 0.1, 0, s.hidden, s.inter)
	}
	config := fmt.Sprintf(`{"model_type":"qwen3","hidden_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,
"num_key_value_heads":%d,"head_dim":%d,"intermediate_size":%d,"vocab_size":%d,"max_position_embeddings":%d,
"rms_norm_eps":1e-6,"rope_theta":1000000,"rope_scaling":null,"attention_bias":false,"hidden_act":"silu"}`,
		s.hidden, s.layers, s.heads, s.kvHeads, s.headDim, s.inter, s.vocab, s.maxPos)
	if err := os.WriteFile(filepath.Join(ck.dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	type entry struct {
		DType   string  `json:"dtype"`
		Shape   []int   `json:"shape"`
		Offsets []int64 `json:"data_offsets"`
	}
	names := make([]string, 0, len(ck.tensors))
	for name := range ck.tensors {
		names = append(names, name)
	}
	slices.Sort(names)
	header := map[string]entry{}
	var data []byte
	for _, name := range names {
		start := int64(len(data))
		for _, v := range ck.tensors[name] {
			data = binary.LittleEndian.AppendUint16(data, uint16(math.Float32bits(v)>>16))
		}
		header[name] = entry{"BF16", ck.shapes[name], []int64{start, int64(len(data))}}
	}
	raw, err := vibejson.Marshal(&header)
	if err != nil {
		t.Fatal(err)
	}
	for len(raw)%8 != 0 {
		raw = append(raw, ' ')
	}
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(raw)))
	file = append(append(file, raw...), data...)
	if err := os.WriteFile(filepath.Join(ck.dir, "model.safetensors"), file, 0o644); err != nil {
		t.Fatal(err)
	}
	return ck
}

func bf16Round(v float32) float32 {
	b := math.Float32bits(v)
	b += 0x7fff + (b>>16)&1
	return math.Float32frombits(b &^ 0xffff)
}

// referenceHidden runs a direct float64 Qwen3 forward pass over the BF16
// checkpoint values and returns the final-normalized last-token state.
func (ck *tinyCheckpoint) referenceHidden(ids []int) []float32 {
	s := tinyShape
	w := func(name string) []float32 { return ck.tensors[name] }
	matvec := func(m []float32, x []float64, n, k int) []float64 {
		out := make([]float64, n)
		for i := range n {
			for j := range k {
				out[i] += float64(m[i*k+j]) * x[j]
			}
		}
		return out
	}
	norm := func(x []float64, weight []float32) []float64 {
		var ss float64
		for _, v := range x {
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(len(x))+1e-6)
		out := make([]float64, len(x))
		for i, v := range x {
			out[i] = v * inv * float64(weight[i])
		}
		return out
	}
	rope := func(v []float64, pos int) {
		half := s.headDim / 2
		for d := range half {
			theta := float64(pos) / math.Pow(1e6, float64(2*d)/float64(s.headDim))
			c, sn := math.Cos(theta), math.Sin(theta)
			a, b := v[d], v[d+half]
			v[d], v[d+half] = a*c-b*sn, b*c+a*sn
		}
	}
	n := len(ids)
	h := make([][]float64, n)
	for i, id := range ids {
		h[i] = make([]float64, s.hidden)
		for j := range s.hidden {
			h[i][j] = float64(w("model.embed_tokens.weight")[id*s.hidden+j])
		}
	}
	qd, kd := s.heads*s.headDim, s.kvHeads*s.headDim
	for l := range s.layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		q, k, v := make([][]float64, n), make([][]float64, n), make([][]float64, n)
		for i := range n {
			x := norm(h[i], w(p+"input_layernorm.weight"))
			q[i] = matvec(w(p+"self_attn.q_proj.weight"), x, qd, s.hidden)
			k[i] = matvec(w(p+"self_attn.k_proj.weight"), x, kd, s.hidden)
			v[i] = matvec(w(p+"self_attn.v_proj.weight"), x, kd, s.hidden)
			for hh := range s.heads {
				copy(q[i][hh*s.headDim:], norm(q[i][hh*s.headDim:(hh+1)*s.headDim], w(p+"self_attn.q_norm.weight")))
				rope(q[i][hh*s.headDim:(hh+1)*s.headDim], i)
			}
			for hh := range s.kvHeads {
				copy(k[i][hh*s.headDim:], norm(k[i][hh*s.headDim:(hh+1)*s.headDim], w(p+"self_attn.k_norm.weight")))
				rope(k[i][hh*s.headDim:(hh+1)*s.headDim], i)
			}
		}
		for i := range n {
			ctx := make([]float64, qd)
			for hh := range s.heads {
				g := hh / (s.heads / s.kvHeads)
				scores := make([]float64, i+1)
				maxScore := math.Inf(-1)
				for j := range i + 1 {
					for d := range s.headDim {
						scores[j] += q[i][hh*s.headDim+d] * k[j][g*s.headDim+d]
					}
					scores[j] /= math.Sqrt(float64(s.headDim))
					maxScore = max(maxScore, scores[j])
				}
				var sum float64
				for j := range scores {
					scores[j] = math.Exp(scores[j] - maxScore)
					sum += scores[j]
				}
				for j := range scores {
					for d := range s.headDim {
						ctx[hh*s.headDim+d] += scores[j] / sum * v[j][g*s.headDim+d]
					}
				}
			}
			o := matvec(w(p+"self_attn.o_proj.weight"), ctx, s.hidden, qd)
			for j := range o {
				h[i][j] += o[j]
			}
			x := norm(h[i], w(p+"post_attention_layernorm.weight"))
			gate := matvec(w(p+"mlp.gate_proj.weight"), x, s.inter, s.hidden)
			up := matvec(w(p+"mlp.up_proj.weight"), x, s.inter, s.hidden)
			for j := range gate {
				gate[j] = gate[j] / (1 + math.Exp(-gate[j])) * up[j]
			}
			down := matvec(w(p+"mlp.down_proj.weight"), gate, s.hidden, s.inter)
			for j := range down {
				h[i][j] += down[j]
			}
		}
	}
	out := norm(h[n-1], w("model.norm.weight"))
	res := make([]float32, len(out))
	for i, v := range out {
		res[i] = float32(v)
	}
	return res
}

func vectorParity(a, b []float32) (cosine, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
		maxAbs = max(maxAbs, math.Abs(float64(a[i]-b[i])))
	}
	return dot / math.Sqrt(na*nb), maxAbs
}

func TestLoadModelExactF16Weights(t *testing.T) {
	ck := writeTinyCheckpoint(t, 1)
	m, err := LoadWeights(ck.dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if m.Format() != WeightsF16 {
		t.Fatalf("default format %q", m.Format())
	}
	// Every BF16 projection weight must round-trip exactly through packing.
	want := ck.tensors["model.layers.1.mlp.down_proj.weight"]
	w := m.layers[1].down
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
	for i, v := range ck.tensors["model.norm.weight"] {
		if m.finalNorm[i] != v {
			t.Fatalf("final norm %d = %g, want %g", i, m.finalNorm[i], v)
		}
	}
}

func TestLoadModelRejectsBadCheckpoints(t *testing.T) {
	ck := writeTinyCheckpoint(t, 2)
	if _, err := LoadWeights(ck.dir, "int4"); err == nil {
		t.Fatal("accepted an unknown weight format")
	}
	cfg := filepath.Join(ck.dir, "config.json")
	raw, _ := os.ReadFile(cfg)
	for _, bad := range []string{`"rope_scaling":{"type":"yarn"}`, `"attention_bias":true`} {
		patched := []byte(string(raw[:len(raw)-1]) + "," + bad + "}")
		// Later duplicate keys are rejected or override; either way the load must fail.
		if err := os.WriteFile(cfg, patched, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadWeights(ck.dir, ""); err == nil {
			t.Fatalf("accepted config with %s", bad)
		}
	}
	if err := os.WriteFile(cfg, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(filepath.Join(ck.dir, "model.safetensors"), 4096); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadWeights(ck.dir, ""); err == nil {
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
	ck := writeTinyCheckpoint(t, 3)
	long := func(n, seed int) []int {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7 + seed) % tinyShape.vocab
		}
		return ids
	}
	// 70 and 290 tokens take the blocked GEMM attention path (three blocks,
	// the last partial); the rest use the streaming path in the same batch.
	seqs := [][]int{{1}, {2, 3, 0, 1, 2}, long(70, 3), {3, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 1, 3, 2, 22, 7, 9}, {0, 2}, long(290, 5), {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
	want := make([][]float32, len(seqs))
	for i, ids := range seqs {
		want[i] = ck.referenceHidden(ids)
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
		m, err := LoadWeights(ck.dir, tc.format)
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
				got[i] = make([]float32, tinyShape.hidden)
			}
			if err := e.HiddenLastBatchInto(seqs, got, ws); err != nil {
				t.Fatal(err)
			}
			single := make([]float32, tinyShape.hidden)
			for i := range seqs {
				cos, maxAbs := vectorParity(got[i], want[i])
				if cos < tc.cosine || maxAbs > tc.abs64 {
					t.Fatalf("%s workers=%d seq %d: cosine=%.9f max_abs=%g", tc.format, workers, i, cos, maxAbs)
				}
				if err := e.HiddenLastInto(seqs[i], single, ws); err != nil {
					t.Fatal(err)
				}
				if cos, maxAbs := vectorParity(single, got[i]); cos < 0.9999999 || maxAbs > 1e-4 {
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
	ck := writeTinyCheckpoint(t, 4)
	m, err := LoadWeights(ck.dir, "")
	if err != nil {
		t.Fatal(err)
	}
	e, _ := NewEvaluator(m)
	ws, _ := e.NewWorkspace(2)
	defer ws.Close()
	dst := make([]float32, tinyShape.hidden)
	for name, ids := range map[string][]int{
		"empty": {}, "negative": {1, -1}, "vocab": {tinyShape.vocab}, "context": make([]int, tinyShape.maxPos+1),
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
	ck := writeTinyCheckpoint(t, 6)
	for _, format := range []string{WeightsF16, WeightsInt8} {
		m, err := LoadWeights(ck.dir, format)
		if err != nil {
			t.Fatal(err)
		}
		e, _ := NewEvaluator(m)
		ws, _ := e.NewWorkspace(3)
		kv, err := e.NewPrefixKV(tinyShape.maxPos)
		if err != nil {
			t.Fatal(err)
		}
		seq := func(n, seed int) []int {
			ids := make([]int, n)
			for i := range ids {
				ids[i] = (i*5 + seed + i/7) % tinyShape.vocab
			}
			return ids
		}
		a := seq(230, 1)
		b := append(append([]int(nil), a[:90]...), seq(40, 9)...) // branches after 90 tokens
		got, fresh := make([]float32, tinyShape.hidden), make([]float32, tinyShape.hidden)
		check := func(name string, full []int) {
			t.Helper()
			if err := e.HiddenLastInto(full, fresh, ws); err != nil {
				t.Fatal(err)
			}
			// Extensions always use blocked attention while short fresh
			// sequences stream, so summation order (and an occasional FP16
			// activation rounding) differs.
			if cos, maxAbs := vectorParity(got, fresh); cos < 0.9999999 || maxAbs > 2e-3 {
				t.Fatalf("%s %s: extension vs fresh cosine=%.9f max_abs=%g", format, name, cos, maxAbs)
			}
			refGate := 0.99999
			if format == WeightsInt8 {
				refGate = 0.999 // per-row int8 weights are lossy by design
			}
			if cos, _ := vectorParity(got, ck.referenceHidden(full)); cos < refGate {
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
		if err := e.HiddenLastExtendInto(kv, 0, seq(tinyShape.maxPos+1, 0), got, ws); err == nil {
			t.Fatal("accepted more tokens than the prefix capacity")
		}
		_ = ws.Close()
	}
}
