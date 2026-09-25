// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/thesyncim/vibejson"
)

// Weight formats accepted by LoadWeights.
const (
	// WeightsF16 stores every BF16 checkpoint weight exactly: each output row
	// is shifted by a power of two into FP16 range. It is the default.
	WeightsF16 = "f16"
	// WeightsInt8 rotates each projection's input space with a randomized
	// Hadamard transform, stores each output row as symmetric int8 with one
	// FP32 scale, and quantizes activations to int8 per row at run time. It
	// halves weight memory and runs the int8 matrix units at about twice the
	// FP16 rate, at a measurable accuracy cost.
	WeightsInt8 = "int8"
	// WeightsGPU runs the model on the Apple GPU through Metal (darwin/arm64
	// only). Projections are rotated like WeightsInt8 and stored as int8 per
	// row; activations stay in FP32, so only the weights are quantized.
	WeightsGPU = "gpu"
	// WeightsGPUQ4 is WeightsGPU with 4-bit weights in blocks of 32 values
	// sharing an FP16 scale: 4.5 bits per weight, for the lowest latency at
	// a measurable accuracy cost.
	WeightsGPUQ4 = "gpu-q4"
)

// Weights is an immutable Qwen3 dense decoder prepared for last-hidden-state
// inference on the CPU. Projections are packed once for the SME tile kernel;
// the language-model head is not loaded. Weights may be shared by any number
// of evaluators and workspaces.
type Weights struct {
	cfg       modelConfig
	embed     []uint16 // BF16 bits, [vocab][hidden]
	finalNorm []float32
	layers    []modelLayer
	format    string
	gpu       *gpuModel // set for WeightsGPU
}

type modelLayer struct {
	attnNorm, mlpNorm, qNorm, kNorm []float32
	q, k, v, o, gate, up, down      linear
}

// linear is one packed projection: exact FP16 weights, or rotated int8
// weights with the rotation their inputs need.
type linear struct {
	f16 *q8gemm.Weights
	i8  *q8gemm.WeightsI8
	rot *rotation
}

func (l *linear) dims() (k, n int) {
	if l.i8 != nil {
		return l.i8.Dims()
	}
	return l.f16.Dims()
}

func (l *linear) panels() int {
	if l.i8 != nil {
		return l.i8.Panels()
	}
	return l.f16.Panels()
}

func (l *linear) bytes() int {
	if l.i8 != nil {
		return l.i8.Bytes()
	}
	return l.f16.Bytes()
}

type modelConfig struct {
	hidden, layers, heads, kvHeads, headDim, kvDim int
	intermediate, vocab, maxPositions              int
	eps, attnScale                                 float64
	invFreq                                        []float64
}

type hfConfig struct {
	ModelType        string  `json:"model_type"`
	HiddenSize       int     `json:"hidden_size"`
	Layers           int     `json:"num_hidden_layers"`
	Heads            int     `json:"num_attention_heads"`
	KVHeads          int     `json:"num_key_value_heads"`
	HeadDim          int     `json:"head_dim"`
	Intermediate     int     `json:"intermediate_size"`
	Vocab            int     `json:"vocab_size"`
	MaxPositions     int     `json:"max_position_embeddings"`
	RMSNormEps       float64 `json:"rms_norm_eps"`
	RopeTheta        float64 `json:"rope_theta"`
	RopeScaling      any     `json:"rope_scaling"`
	AttentionBias    bool    `json:"attention_bias"`
	HiddenAct        string  `json:"hidden_act"`
	UseSlidingWindow bool    `json:"use_sliding_window"`
}

// LoadWeights reads an official Qwen3 safetensors snapshot directory
// (config.json plus model.safetensors or a sharded index). format is one of
// the Weights constants, or empty to choose the fastest backend available:
// WeightsGPU for the Qwen3-8B geometry on a Metal GPU, WeightsF16 otherwise.
// Tensors are read and packed in parallel; peak memory is the packed model
// plus one tensor per loader.
func LoadWeights(dir, format string) (*Weights, error) {
	if format == "" {
		format = WeightsF16
		if cfg, err := readConfig(filepath.Join(dir, "config.json")); err == nil && gpuSupports(&cfg) {
			format = WeightsGPU
		}
	}
	if format != WeightsF16 && format != WeightsInt8 && format != WeightsGPU && format != WeightsGPUQ4 {
		return nil, fmt.Errorf("qwen3: unsupported weight format %q", format)
	}
	cfg, err := readConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	st, err := safetensors.Open(dir)
	if err != nil {
		return nil, err
	}
	defer st.Close()

	m := &Weights{cfg: cfg, format: format, layers: make([]modelLayer, cfg.layers)}
	h, kv, inter := cfg.hidden, cfg.kvDim, cfg.intermediate
	qdim := cfg.heads * cfg.headDim
	if m.embed, err = st.BF16("model.embed_tokens.weight", cfg.vocab, h); err != nil {
		return nil, err
	}
	if m.finalNorm, err = st.Float32("model.norm.weight", h); err != nil {
		return nil, err
	}
	var rotHidden, rotContext, rotInter *rotation
	if format == WeightsInt8 {
		rotHidden, rotContext, rotInter = newRotation(h), newRotation(qdim), newRotation(inter)
	}
	type job struct {
		name string
		n, k int
		dst  *linear
		rot  *rotation
	}
	var jobs []job
	for i := range m.layers {
		l := &m.layers[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		for _, v := range []struct {
			name string
			n    int
			dst  *[]float32
		}{
			{"input_layernorm.weight", h, &l.attnNorm},
			{"post_attention_layernorm.weight", h, &l.mlpNorm},
			{"self_attn.q_norm.weight", cfg.headDim, &l.qNorm},
			{"self_attn.k_norm.weight", cfg.headDim, &l.kNorm},
		} {
			if *v.dst, err = st.Float32(p+v.name, v.n); err != nil {
				return nil, err
			}
		}
		jobs = append(jobs,
			job{p + "self_attn.q_proj.weight", qdim, h, &l.q, rotHidden},
			job{p + "self_attn.k_proj.weight", kv, h, &l.k, rotHidden},
			job{p + "self_attn.v_proj.weight", kv, h, &l.v, rotHidden},
			job{p + "self_attn.o_proj.weight", h, qdim, &l.o, rotContext},
			job{p + "mlp.gate_proj.weight", inter, h, &l.gate, rotHidden},
			job{p + "mlp.up_proj.weight", inter, h, &l.up, rotHidden},
			job{p + "mlp.down_proj.weight", h, inter, &l.down, rotInter},
		)
	}
	if format == WeightsGPU || format == WeightsGPUQ4 {
		bits := 8
		if format == WeightsGPUQ4 {
			bits = 4
		}
		if err := m.loadGPU(st, bits); err != nil {
			return nil, err
		}
		return m, nil
	}
	workers := min(runtime.GOMAXPROCS(0), 8, len(jobs))
	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		first   error
		next    int
		maxSize = max(inter*h, qdim*h)
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]uint16, maxSize)
			for {
				mu.Lock()
				if first != nil || next == len(jobs) {
					mu.Unlock()
					return
				}
				j := jobs[next]
				next++
				mu.Unlock()
				w, err := loadProjection(st, j.name, j.n, j.k, buf, j.rot)
				mu.Lock()
				if err != nil && first == nil {
					first = err
				}
				*j.dst = w
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if first != nil {
		return nil, first
	}
	return m, nil
}

// Config is the geometry of a Qwen3 model.
type Config struct {
	Hidden, Layers, Heads, KVHeads, HeadDim, Intermediate, Vocab, MaxPositions int
}

// Config reports the model's geometry.
func (m *Weights) Config() Config {
	c := &m.cfg
	return Config{Hidden: c.hidden, Layers: c.layers, Heads: c.heads, KVHeads: c.kvHeads, HeadDim: c.headDim,
		Intermediate: c.intermediate, Vocab: c.vocab, MaxPositions: c.maxPositions}
}

// Format reports the projection weight format.
func (m *Weights) Format() string { return m.format }

// WeightBytes reports the resident bytes of packed projection weights.
func (m *Weights) WeightBytes() int64 {
	var n int64
	for i := range m.layers {
		l := &m.layers[i]
		for _, w := range [...]*linear{&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down} {
			n += int64(w.bytes())
		}
	}
	return n
}

func (m *Weights) embedRow(id int, dst []float32) {
	row := m.embed[id*m.cfg.hidden : (id+1)*m.cfg.hidden]
	for i, b := range row {
		dst[i] = q8gemm.BF16ToF32(b)
	}
}

func readConfig(path string) (modelConfig, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return modelConfig{}, fmt.Errorf("qwen3: read Qwen3 config: %w", err)
	}
	var c hfConfig
	if err := vibejson.Unmarshal(raw, &c); err != nil {
		return modelConfig{}, fmt.Errorf("qwen3: parse Qwen3 config: %w", err)
	}
	if c.ModelType != "qwen3" {
		return modelConfig{}, fmt.Errorf("qwen3: expected model_type qwen3, got %q", c.ModelType)
	}
	if c.HiddenAct != "" && c.HiddenAct != "silu" || c.AttentionBias || c.UseSlidingWindow || c.RopeScaling != nil {
		return modelConfig{}, errors.New("qwen3: unsupported Qwen3 variant (activation, attention bias, sliding window, or scaled RoPE)")
	}
	if c.HeadDim == 0 && c.Heads > 0 {
		c.HeadDim = c.HiddenSize / c.Heads
	}
	const limit = 1 << 20
	for _, v := range []int{c.HiddenSize, c.Layers, c.Heads, c.KVHeads, c.HeadDim, c.Intermediate, c.Vocab, c.MaxPositions} {
		if v <= 0 || v > limit*16 {
			return modelConfig{}, errors.New("qwen3: incomplete or implausible Qwen3 geometry")
		}
	}
	if c.Heads%c.KVHeads != 0 || c.HeadDim%2 != 0 || c.RMSNormEps <= 0 || c.RopeTheta <= 0 {
		return modelConfig{}, errors.New("qwen3: inconsistent Qwen3 attention, normalization, or RoPE configuration")
	}
	inv := make([]float64, c.HeadDim/2)
	for d := range inv {
		inv[d] = 1 / math.Pow(c.RopeTheta, float64(2*d)/float64(c.HeadDim))
	}
	return modelConfig{
		hidden: c.HiddenSize, layers: c.Layers, heads: c.Heads, kvHeads: c.KVHeads,
		headDim: c.HeadDim, kvDim: c.KVHeads * c.HeadDim, intermediate: c.Intermediate,
		vocab: c.Vocab, maxPositions: c.MaxPositions, eps: c.RMSNormEps,
		attnScale: 1 / math.Sqrt(float64(c.HeadDim)), invFreq: inv,
	}, nil
}

// loadProjection reads an [n][k] BF16 matrix into buf and packs it: exactly as
// FP16 when rot is nil, otherwise as rotated per-row int8.
func loadProjection(st *safetensors.Checkpoint, name string, n, k int, buf []uint16, rot *rotation) (linear, error) {
	t, err := st.Lookup(name, n, k)
	if err != nil {
		return linear{}, fmt.Errorf("qwen3: %w", err)
	}
	if t.DType != "BF16" {
		return linear{}, fmt.Errorf("qwen3: %s is %s; the loader expects the official BF16 checkpoint", name, t.DType)
	}
	raw := buf[:n*k]
	if err := t.ReadBits(raw, 0); err != nil {
		return linear{}, fmt.Errorf("qwen3: %w", err)
	}
	if rot == nil {
		w, err := q8gemm.NewWeightsF16(k, n)
		if err != nil {
			return linear{}, err
		}
		if _, err := w.PackBF16(raw); err != nil {
			return linear{}, fmt.Errorf("qwen3: pack %s: %w", name, err)
		}
		return linear{f16: w}, nil
	}
	q, scales := quantizeRotatedRows(raw, n, k, rot)
	w, err := q8gemm.NewWeightsI8(k, n)
	if err != nil {
		return linear{}, err
	}
	if err := w.Pack(q, scales); err != nil {
		return linear{}, fmt.Errorf("qwen3: pack %s: %w", name, err)
	}
	return linear{i8: w, rot: rot}, nil
}

// quantizeRotatedRows rotates each weight row by rot and maps it to
// round(w/s) with s = max|w|/127.
func quantizeRotatedRows(bf16 []uint16, n, k int, rot *rotation) ([]int8, []float32) {
	q := make([]int8, n*k)
	scales := make([]float32, n)
	row := make([]float32, k)
	for r := range n {
		for i, b := range bf16[r*k : (r+1)*k] {
			row[i] = q8gemm.BF16ToF32(b)
		}
		rot.apply(row)
		m := q8gemm.MaxAbs(row)
		if m == 0 {
			continue
		}
		scale := m / 127
		scales[r] = scale
		for i, v := range row {
			q[r*k+i] = int8(max(-127, min(127, math.RoundToEven(float64(v/scale)))))
		}
	}
	return q, scales
}
