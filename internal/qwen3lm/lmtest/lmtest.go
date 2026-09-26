// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package lmtest writes small random Qwen3 checkpoints and evaluates them
// with a direct float64 forward pass, as the oracle for the tests of every
// package built on internal/qwen3lm.
package lmtest

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/thesyncim/vibejson"
)

// Shape is a small Qwen3 geometry that still exercises grouped-query
// attention, partial output panels, odd tile counts, and attention wider
// than the hidden state (as in Qwen3-0.6B and Qwen3-4B).
var Shape = struct {
	Hidden, Layers, Heads, KVHeads, HeadDim, Inter, Vocab, MaxPos int
}{Hidden: 96, Layers: 2, Heads: 8, KVHeads: 2, HeadDim: 24, Inter: 136, Vocab: 23, MaxPos: 320}

// Checkpoint is a random checkpoint written by Write.
type Checkpoint struct {
	Dir     string
	Tensors map[string][]float32 // BF16-rounded values
	Shapes  map[string][]int
}

// Write writes a random Qwen3 snapshot (config and one
// safetensors file) whose values are exactly representable in BF16.
func Write(t testing.TB, seed int64) *Checkpoint {
	t.Helper()
	return WriteNamed(t, seed, nil)
}

// WriteNamed is Write with each tensor stored under rename(name), as a model
// that nests the decoder stores it. Tensors keeps the Qwen3 names.
func WriteNamed(t testing.TB, seed int64, rename func(string) string) *Checkpoint {
	t.Helper()
	if rename == nil {
		rename = func(name string) string { return name }
	}
	s := Shape
	rng := rand.New(rand.NewSource(seed))
	ck := &Checkpoint{Dir: t.TempDir(), Tensors: map[string][]float32{}, Shapes: map[string][]int{}}
	add := func(name string, scale float64, offset float64, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = BF16Round(float32(offset + rng.NormFloat64()*scale))
		}
		ck.Tensors[name], ck.Shapes[name] = v, shape
	}
	add("model.embed_tokens.weight", 1, 0, s.Vocab, s.Hidden)
	add("model.norm.weight", 0.1, 1, s.Hidden)
	add("lm_head.weight", 0.02, 0, s.Vocab, s.Hidden)
	for l := range s.Layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		add(p+"input_layernorm.weight", 0.1, 1, s.Hidden)
		add(p+"post_attention_layernorm.weight", 0.1, 1, s.Hidden)
		add(p+"self_attn.q_norm.weight", 0.1, 1, s.HeadDim)
		add(p+"self_attn.k_norm.weight", 0.1, 1, s.HeadDim)
		add(p+"self_attn.q_proj.weight", 0.1, 0, s.Heads*s.HeadDim, s.Hidden)
		add(p+"self_attn.k_proj.weight", 0.1, 0, s.KVHeads*s.HeadDim, s.Hidden)
		add(p+"self_attn.v_proj.weight", 0.1, 0, s.KVHeads*s.HeadDim, s.Hidden)
		add(p+"self_attn.o_proj.weight", 0.1, 0, s.Hidden, s.Heads*s.HeadDim)
		add(p+"mlp.gate_proj.weight", 0.1, 0, s.Inter, s.Hidden)
		add(p+"mlp.up_proj.weight", 0.1, 0, s.Inter, s.Hidden)
		add(p+"mlp.down_proj.weight", 0.1, 0, s.Hidden, s.Inter)
	}
	config := fmt.Sprintf(`{"model_type":"qwen3","hidden_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,
"num_key_value_heads":%d,"head_dim":%d,"intermediate_size":%d,"vocab_size":%d,"max_position_embeddings":%d,
"rms_norm_eps":1e-6,"rope_theta":1000000,"rope_scaling":null,"attention_bias":false,"hidden_act":"silu"}`,
		s.Hidden, s.Layers, s.Heads, s.KVHeads, s.HeadDim, s.Inter, s.Vocab, s.MaxPos)
	if err := os.WriteFile(filepath.Join(ck.Dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	type entry struct {
		DType   string  `json:"dtype"`
		Shape   []int   `json:"shape"`
		Offsets []int64 `json:"data_offsets"`
	}
	names := make([]string, 0, len(ck.Tensors))
	for name := range ck.Tensors {
		names = append(names, name)
	}
	slices.Sort(names)
	header := map[string]entry{}
	var data []byte
	for _, name := range names {
		start := int64(len(data))
		for _, v := range ck.Tensors[name] {
			data = binary.LittleEndian.AppendUint16(data, uint16(math.Float32bits(v)>>16))
		}
		header[rename(name)] = entry{"BF16", ck.Shapes[name], []int64{start, int64(len(data))}}
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
	if err := os.WriteFile(filepath.Join(ck.Dir, "model.safetensors"), file, 0o644); err != nil {
		t.Fatal(err)
	}
	return ck
}

// BF16Round rounds v to the nearest BF16 value, ties to even.
func BF16Round(v float32) float32 {
	b := math.Float32bits(v)
	b += 0x7fff + (b>>16)&1
	return math.Float32frombits(b &^ 0xffff)
}

// ReferenceHidden runs a direct float64 Qwen3 forward pass over the BF16
// checkpoint values and returns the final-normalized last-token state.
func (ck *Checkpoint) ReferenceHidden(ids []int) []float32 {
	return ck.ReferenceHiddenEmbeds(ids, -1, nil)
}

// ReferenceHiddenEmbeds is ReferenceHidden with the i-th occurrence of
// token taking row i of rows as its input embedding.
func (ck *Checkpoint) ReferenceHiddenEmbeds(ids []int, token int, rows []float32) []float32 {
	hidden, _ := ck.forward(ids, token, rows, -1, -1)
	return hidden
}

// ReferenceAttention returns the attention probabilities of the last of
// ids over every position, in layer layer and query head head.
func (ck *Checkpoint) ReferenceAttention(ids []int, layer, head int) []float64 {
	_, probs := ck.forward(ids, -1, nil, layer, head)
	return probs
}

// forward evaluates ids, returning the last position's final-normalized
// state and, when layer is not negative, its attention probabilities in
// that layer and query head.
func (ck *Checkpoint) forward(ids []int, token int, rows []float32, layer, head int) ([]float32, []float64) {
	s := Shape
	var probs []float64
	w := func(name string) []float32 { return ck.Tensors[name] }
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
		half := s.HeadDim / 2
		for d := range half {
			theta := float64(pos) / math.Pow(1e6, float64(2*d)/float64(s.HeadDim))
			c, sn := math.Cos(theta), math.Sin(theta)
			a, b := v[d], v[d+half]
			v[d], v[d+half] = a*c-b*sn, b*c+a*sn
		}
	}
	n := len(ids)
	h := make([][]float64, n)
	spliced := 0
	for i, id := range ids {
		h[i] = make([]float64, s.Hidden)
		src := w("model.embed_tokens.weight")[id*s.Hidden:]
		if id == token {
			src = rows[spliced*s.Hidden:]
			spliced++
		}
		for j := range s.Hidden {
			h[i][j] = float64(src[j])
		}
	}
	qd, kd := s.Heads*s.HeadDim, s.KVHeads*s.HeadDim
	for l := range s.Layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		q, k, v := make([][]float64, n), make([][]float64, n), make([][]float64, n)
		for i := range n {
			x := norm(h[i], w(p+"input_layernorm.weight"))
			q[i] = matvec(w(p+"self_attn.q_proj.weight"), x, qd, s.Hidden)
			k[i] = matvec(w(p+"self_attn.k_proj.weight"), x, kd, s.Hidden)
			v[i] = matvec(w(p+"self_attn.v_proj.weight"), x, kd, s.Hidden)
			for hh := range s.Heads {
				copy(q[i][hh*s.HeadDim:], norm(q[i][hh*s.HeadDim:(hh+1)*s.HeadDim], w(p+"self_attn.q_norm.weight")))
				rope(q[i][hh*s.HeadDim:(hh+1)*s.HeadDim], i)
			}
			for hh := range s.KVHeads {
				copy(k[i][hh*s.HeadDim:], norm(k[i][hh*s.HeadDim:(hh+1)*s.HeadDim], w(p+"self_attn.k_norm.weight")))
				rope(k[i][hh*s.HeadDim:(hh+1)*s.HeadDim], i)
			}
		}
		for i := range n {
			ctx := make([]float64, qd)
			for hh := range s.Heads {
				g := hh / (s.Heads / s.KVHeads)
				scores := make([]float64, i+1)
				maxScore := math.Inf(-1)
				for j := range i + 1 {
					for d := range s.HeadDim {
						scores[j] += q[i][hh*s.HeadDim+d] * k[j][g*s.HeadDim+d]
					}
					scores[j] /= math.Sqrt(float64(s.HeadDim))
					maxScore = max(maxScore, scores[j])
				}
				var sum float64
				for j := range scores {
					scores[j] = math.Exp(scores[j] - maxScore)
					sum += scores[j]
				}
				if l == layer && hh == head && i == n-1 {
					probs = make([]float64, len(scores))
					for j, v := range scores {
						probs[j] = v / sum
					}
				}
				for j := range scores {
					for d := range s.HeadDim {
						ctx[hh*s.HeadDim+d] += scores[j] / sum * v[j][g*s.HeadDim+d]
					}
				}
			}
			o := matvec(w(p+"self_attn.o_proj.weight"), ctx, s.Hidden, qd)
			for j := range o {
				h[i][j] += o[j]
			}
			x := norm(h[i], w(p+"post_attention_layernorm.weight"))
			gate := matvec(w(p+"mlp.gate_proj.weight"), x, s.Inter, s.Hidden)
			up := matvec(w(p+"mlp.up_proj.weight"), x, s.Inter, s.Hidden)
			for j := range gate {
				gate[j] = gate[j] / (1 + math.Exp(-gate[j])) * up[j]
			}
			down := matvec(w(p+"mlp.down_proj.weight"), gate, s.Hidden, s.Inter)
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
	return res, probs
}

// ReferenceLogits applies lm_head.weight to a final-normalized state in
// float64.
func (ck *Checkpoint) ReferenceLogits(hidden []float32) []float32 {
	s := Shape
	head := ck.Tensors["lm_head.weight"]
	out := make([]float32, s.Vocab)
	for i := range out {
		var sum float64
		for j, v := range hidden {
			sum += float64(head[i*s.Hidden+j]) * float64(v)
		}
		out[i] = float32(sum)
	}
	return out
}

// VectorParity returns the cosine similarity and largest absolute
// difference of a and b.
func VectorParity(a, b []float32) (cosine, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
		maxAbs = max(maxAbs, math.Abs(float64(a[i]-b[i])))
	}
	return dot / math.Sqrt(na*nb), maxAbs
}
