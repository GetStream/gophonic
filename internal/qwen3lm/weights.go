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

	"github.com/GetStream/gophonic/internal/arena"
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
	// WeightsGPUQ8 is WeightsGPU with int8 weights in blocks of 32 values
	// sharing an FP16 scale, as GGML's Q8_0 on top of the rotation: 8.5 bits
	// per weight, for models (such as Qwen3-1.7B) that lose precision with a
	// single scale per row.
	WeightsGPUQ8 = "gpu-q8"
	// WeightsGPUQ4 is WeightsGPU with 4-bit weights in blocks of 32 values
	// sharing an FP16 scale: 4.5 bits per weight, for the lowest latency at
	// a measurable accuracy cost.
	WeightsGPUQ4 = "gpu-q4"
)

// Weights is an immutable Qwen3 dense decoder prepared for last-hidden-state
// inference on the CPU. Projections are packed once for the SME tile kernel;
// the language-model head is loaded only when LoadOptions.Head names it.
// Weights may be shared by any number of evaluators and workspaces.
type Weights struct {
	memory    *arena.Arena
	cfg       modelConfig
	embed     []uint16 // BF16 bits, [vocab][hidden]
	finalNorm []float32
	layers    []modelLayer
	head      linear // language-model head, [vocab][hidden]; unset unless loaded
	prefix    string
	format    string
	gpu       *gpuModel      // set for WeightsGPU
	unmap     []func() error // mappings to release
}

type modelLayer struct {
	attnNorm, mlpNorm, qNorm, kNorm []float32
	q, k, v, o, gate, up, down      linear
	// A DeltaNet layer's -exp(A_log), time step bias, output norm weight,
	// and causal convolution ([channel][DCONV]).
	dnNegA, dnDT, dnNorm, dnConv []float32
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
	// A mixture of experts (Qwen3-MoE) replaces every layer's MLP with
	// experts of intermediate width, of which topK take each token, weighted
	// by the router's renormalized probabilities.
	experts, topK int
	// The Qwen3.5 family is hybrid: most layers mix tokens with a Gated
	// DeltaNet (linear[i]) instead of attention, attention layers gate their
	// output and rotate only the first rotary dimensions of each head, norms
	// scale by 1 + w, and a shared expert joins the routed ones.
	hybrid                   bool
	linear                   []bool
	rotary                   int
	dnKeyHeads, dnValueHeads int // DeltaNet heads: keys (and queries), values
	dnKeyDim, dnValueDim     int
	dnConv                   int // causal convolution width
}

// shared counts the experts every token uses besides its routed ones: the
// Qwen3.5 family's shared expert, stored as expert E.
func (c *modelConfig) shared() int {
	if c.hybrid {
		return 1
	}
	return 0
}

// slots is how many experts each token runs, and allExperts how many are
// stored.
func (c *modelConfig) slots() int      { return c.topK + c.shared() }
func (c *modelConfig) allExperts() int { return c.experts + c.shared() }

// routerRows is the FP32 router's rows: the experts', the shared expert's
// gate, and zero rows up to a multiple of 16.
func routerRows(c *modelConfig) int { return (c.experts + c.shared() + 15) / 16 * 16 }

// attnLayers counts the layers that attend (all of them unless hybrid).
func (c *modelConfig) attnLayers() int {
	n := c.layers
	for _, l := range c.linear {
		if l {
			n--
		}
	}
	return n
}

// TextConfig is a Qwen3 decoder's Hugging Face configuration: the
// config.json of a Qwen3 snapshot, or the text_config a multimodal model such
// as Qwen3-ASR nests.
type TextConfig struct {
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
	// Qwen3-MoE (model_type qwen3_moe).
	Experts         int   `json:"num_experts"`
	ExpertsPerToken int   `json:"num_experts_per_tok"`
	ExpertWidth     int   `json:"moe_intermediate_size"`
	NormTopK        bool  `json:"norm_topk_prob"`
	SparseStep      int   `json:"decoder_sparse_step"`
	MLPOnlyLayers   []int `json:"mlp_only_layers"`
	// The Qwen3.5 family (model_type qwen3_5_moe_text).
	LayerTypes        []string        `json:"layer_types"`
	LinearConv        int             `json:"linear_conv_kernel_dim"`
	LinearKeyHeads    int             `json:"linear_num_key_heads"`
	LinearValueHeads  int             `json:"linear_num_value_heads"`
	LinearKeyDim      int             `json:"linear_key_head_dim"`
	LinearValueDim    int             `json:"linear_value_head_dim"`
	AttnOutputGate    bool            `json:"attn_output_gate"`
	PartialRotary     float64         `json:"partial_rotary_factor"`
	RopeParameters    *RopeParameters `json:"rope_parameters"`
	SharedExpertWidth int             `json:"shared_expert_intermediate_size"`
}

// RopeParameters is the rope_parameters of newer configurations. Text uses
// one position for all three multimodal axes, which makes mRoPE plain RoPE.
type RopeParameters struct {
	RopeType      string  `json:"rope_type"`
	RopeTheta     float64 `json:"rope_theta"`
	PartialRotary float64 `json:"partial_rotary_factor"`
}

// LoadWeights reads an official Qwen3 safetensors snapshot directory
// (config.json plus model.safetensors or a sharded index). format is one of
// the Weights constants, or empty to choose the fastest backend that keeps
// Q8_0-level fidelity: on a Metal GPU, WeightsGPU for Qwen3-8B (whose rows
// quantize well with one scale) and WeightsGPUQ8 for other sizes; otherwise
// WeightsF16.
// Tensors are read and packed in parallel; peak memory is the packed model
// plus one tensor per loader.
func LoadWeights(dir, format string) (*Weights, error) {
	return Load(dir, LoadOptions{Format: format})
}

// LoadOptions selects how Load reads a Qwen3 decoder.
type LoadOptions struct {
	// Format is one of the Weights constants, or empty for the fastest
	// faithful backend, as LoadWeights chooses.
	Format string
	// Prefix precedes every decoder tensor name: "model." (the default) for
	// a Qwen3 snapshot, "thinker.model." for Qwen3-ASR.
	Prefix string
	// Config is the decoder configuration. Nil reads config.json in dir.
	Config *TextConfig
	// Head names the language-model head tensor, such as "lm_head.weight",
	// for Logits. Empty loads no head.
	Head string
	// Embed names the token embedding table; empty means Prefix +
	// "embed_tokens.weight".
	Embed string
	// NoEmbed loads no embedding table: every input position is a row of
	// Embeds, as for a model that only ever consumes precomputed inputs.
	NoEmbed bool
}

// headChunkRows overrides the head's row-chunk size in tests.
var headChunkRows int

// Load reads a Qwen3 decoder from the safetensors checkpoint in dir, which
// may hold it inside a larger model.
func Load(dir string, opts LoadOptions) (_ *Weights, err error) {
	var cfg modelConfig
	if opts.Config != nil {
		cfg, err = opts.Config.model()
	} else {
		cfg, err = readConfig(filepath.Join(dir, "config.json"))
	}
	if err != nil {
		return nil, err
	}
	format := opts.Format
	if format == "" {
		format = WeightsF16
		if gpuSupports(&cfg) {
			format = WeightsGPUQ8
			if cfg.hidden == 4096 && cfg.layers == 36 { // Qwen3-8B
				format = WeightsGPU
			}
		}
	}
	if format != WeightsF16 && format != WeightsInt8 && format != WeightsGPU && format != WeightsGPUQ8 && format != WeightsGPUQ4 {
		return nil, fmt.Errorf("qwen3: unsupported weight format %q", format)
	}
	if cfg.experts > 0 && format != WeightsGPUQ8 {
		return nil, fmt.Errorf("qwen3: a mixture of experts runs in the %s format, not %s", WeightsGPUQ8, format)
	}
	prefix := opts.Prefix
	if prefix == "" {
		prefix = "model."
		if cfg.hybrid {
			prefix = "model.language_model." // the multimodal snapshot's decoder
		}
	}
	st, err := safetensors.Open(dir)
	if err != nil {
		return nil, err
	}
	defer st.Close()

	m := &Weights{cfg: cfg, format: format, prefix: prefix, layers: make([]modelLayer, cfg.layers)}
	defer func() {
		if err != nil {
			m.Release()
		}
	}()
	h, kv, inter := cfg.hidden, cfg.kvDim, cfg.intermediate
	qdim := cfg.heads * cfg.headDim
	// The embedding table is used as stored: it is mapped, not read.
	if !opts.NoEmbed {
		name := opts.Embed
		if name == "" {
			name = prefix + "embed_tokens.weight"
		}
		embed, unmap, err := st.MapBF16(name, cfg.vocab, h)
		if err != nil {
			return nil, err
		}
		m.embed, m.unmap = embed, append(m.unmap, unmap)
	}
	if cfg.hybrid {
		if err := m.loadHybrid(st); err != nil {
			return nil, err
		}
		if err := m.loadGPU(st, 9, opts.Head); err != nil {
			return nil, err
		}
		return m, nil
	}
	if m.finalNorm, err = st.Float32(prefix+"norm.weight", h); err != nil {
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
		rows [2]int // for the head: the row range this job packs into dst
	}
	var jobs []job
	for i := range m.layers {
		l := &m.layers[i]
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
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
			job{p + "self_attn.q_proj.weight", qdim, h, &l.q, rotHidden, [2]int{}},
			job{p + "self_attn.k_proj.weight", kv, h, &l.k, rotHidden, [2]int{}},
			job{p + "self_attn.v_proj.weight", kv, h, &l.v, rotHidden, [2]int{}},
			job{p + "self_attn.o_proj.weight", h, qdim, &l.o, rotContext, [2]int{}},
			job{p + "mlp.gate_proj.weight", inter, h, &l.gate, rotHidden, [2]int{}},
			job{p + "mlp.up_proj.weight", inter, h, &l.up, rotHidden, [2]int{}},
			job{p + "mlp.down_proj.weight", h, inter, &l.down, rotInter, [2]int{}},
		)
	}
	if format == WeightsGPU || format == WeightsGPUQ8 || format == WeightsGPUQ4 {
		bits := map[string]int{WeightsGPU: 8, WeightsGPUQ8: 9, WeightsGPUQ4: 4}[format]
		if err := m.loadGPU(st, bits, opts.Head); err != nil {
			return nil, err
		}
		return m, nil
	}
	if format == WeightsF16 {
		sizes := make([]int, len(jobs), len(jobs)+1)
		for i, j := range jobs {
			if sizes[i], err = q8gemm.WeightsF16Bytes(j.k, j.n); err != nil {
				return nil, err
			}
		}
		if opts.Head != "" {
			size, e := q8gemm.WeightsF16Bytes(h, cfg.vocab)
			if e != nil {
				return nil, e
			}
			sizes = append(sizes, size)
		}
		if m.memory, err = arena.NewBytes(sizes...); err != nil {
			return nil, err
		}
		for i, j := range jobs {
			if j.dst.f16, err = q8gemm.NewWeightsF16Buffer(j.k, j.n, m.memory.TakeBytes(sizes[i])); err != nil {
				return nil, err
			}
		}
		if opts.Head != "" {
			if m.head.f16, err = q8gemm.NewWeightsF16Buffer(h, cfg.vocab, m.memory.TakeBytes(sizes[len(jobs)])); err != nil {
				return nil, err
			}
		}
	}
	maxSize := max(inter*h, qdim*h)
	if opts.Head != "" {
		// The head is the largest matrix (vocabulary × hidden); it is read
		// and packed in row chunks no larger than a layer projection.
		if format != WeightsF16 {
			if m.head, err = newHead(&cfg, format, rotHidden); err != nil {
				return nil, err
			}
		}
		step := max(q8gemm.OutputPanel, maxSize/h/q8gemm.OutputPanel*q8gemm.OutputPanel)
		if headChunkRows > 0 {
			step = headChunkRows
		}
		for r := 0; r < cfg.vocab; r += step {
			jobs = append(jobs, job{opts.Head, cfg.vocab, h, &m.head, rotHidden, [2]int{r, min(r+step, cfg.vocab)}})
		}
	}
	workers := min(runtime.GOMAXPROCS(0), 8, len(jobs))
	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		first error
		next  int
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
				var err error
				if j.rows[1] > 0 {
					err = loadRows(st, j.name, j.n, j.k, j.rows[0], j.rows[1], buf, j.dst)
				} else if format == WeightsF16 {
					err = loadRows(st, j.name, j.n, j.k, 0, j.n, buf, j.dst)
				} else {
					var w linear
					w, err = loadProjection(st, j.name, j.n, j.k, buf, j.rot)
					*j.dst = w
				}
				mu.Lock()
				if err != nil && first == nil {
					first = err
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	runtime.KeepAlive(m)
	if first != nil {
		return nil, first
	}
	return m, nil
}

// Release frees what the Weights hold outside the Go heap: GPU buffers and
// mapped files. The Weights are unusable afterwards.
func (m *Weights) Release() {
	_ = m.memory.Close()
	m.memory = nil
	m.releaseGPU()
	for _, unmap := range m.unmap {
		unmap()
	}
	m.unmap, m.embed = nil, nil
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

// GPU reports whether the weights run on the GPU.
func (m *Weights) GPU() bool { return m.gpu != nil }

// WeightBytes reports the resident bytes of packed projection weights,
// including the language-model head when loaded.
func (m *Weights) WeightBytes() int64 {
	var n int64
	if m.head.f16 != nil || m.head.i8 != nil {
		n += int64(m.head.bytes())
	}
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
	var c TextConfig
	if err := vibejson.Unmarshal(raw, &c); err != nil {
		return modelConfig{}, fmt.Errorf("qwen3: parse Qwen3 config: %w", err)
	}
	if c.ModelType == "qwen3_5_moe" {
		// A multimodal snapshot: the decoder is its text_config.
		var outer struct {
			Text TextConfig `json:"text_config"`
		}
		if err := vibejson.Unmarshal(raw, &outer); err != nil {
			return modelConfig{}, fmt.Errorf("qwen3: parse Qwen3.5 config: %w", err)
		}
		c = outer.Text
	}
	return c.model()
}

// model validates the configuration and derives the evaluator's geometry.
func (c TextConfig) model() (modelConfig, error) {
	experts, topK := 0, 0
	var hy modelConfig
	switch c.ModelType {
	case "qwen3_5_moe_text":
		var err error
		if hy, err = c.hybrid(); err != nil {
			return modelConfig{}, err
		}
		experts, topK, c.Intermediate = c.Experts, c.ExpertsPerToken, c.ExpertWidth
	case "qwen3":
	case "qwen3_moe":
		// Every layer sparse, with renormalized top-k weights: Qwen3-30B-A3B
		// and Qwen3-235B-A22B.
		if c.Experts <= 0 || c.ExpertsPerToken <= 0 || c.ExpertsPerToken > c.Experts || c.ExpertWidth <= 0 ||
			!c.NormTopK || c.SparseStep != 1 || len(c.MLPOnlyLayers) != 0 {
			return modelConfig{}, errors.New("qwen3: unsupported Qwen3-MoE layout (dense layers, or experts without renormalized top-k)")
		}
		experts, topK, c.Intermediate = c.Experts, c.ExpertsPerToken, c.ExpertWidth
	default:
		return modelConfig{}, fmt.Errorf("qwen3: expected model_type qwen3, qwen3_moe, or qwen3_5_moe_text, got %q", c.ModelType)
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
	rotary := c.HeadDim
	if hy.hybrid {
		rotary = hy.rotary
	}
	inv := make([]float64, rotary/2)
	for d := range inv {
		inv[d] = 1 / math.Pow(c.RopeTheta, float64(2*d)/float64(rotary))
	}
	return modelConfig{
		hidden: c.HiddenSize, layers: c.Layers, heads: c.Heads, kvHeads: c.KVHeads,
		headDim: c.HeadDim, kvDim: c.KVHeads * c.HeadDim, intermediate: c.Intermediate,
		vocab: c.Vocab, maxPositions: c.MaxPositions, eps: c.RMSNormEps,
		attnScale: 1 / math.Sqrt(float64(c.HeadDim)), invFreq: inv,
		experts: experts, topK: topK,
		hybrid: hy.hybrid, linear: hy.linear, rotary: rotary,
		dnKeyHeads: hy.dnKeyHeads, dnValueHeads: hy.dnValueHeads, dnKeyDim: hy.dnKeyDim, dnValueDim: hy.dnValueDim, dnConv: hy.dnConv,
	}, nil
}

// hybrid validates a Qwen3.5-family configuration: Gated DeltaNet and gated
// attention layers, each with a mixture of experts and a shared expert as
// wide as the routed ones. It fills c's RoPE fields from rope_parameters.
func (c *TextConfig) hybrid() (modelConfig, error) {
	if rp := c.RopeParameters; rp != nil {
		if rp.RopeType != "" && rp.RopeType != "default" {
			return modelConfig{}, fmt.Errorf("qwen3: unsupported RoPE type %q", rp.RopeType)
		}
		c.RopeTheta = rp.RopeTheta
		if rp.PartialRotary != 0 {
			c.PartialRotary = rp.PartialRotary
		}
	}
	if c.PartialRotary == 0 {
		c.PartialRotary = 1
	}
	h := modelConfig{hybrid: true, linear: make([]bool, len(c.LayerTypes)), rotary: int(float64(c.HeadDim) * c.PartialRotary),
		dnKeyHeads: c.LinearKeyHeads, dnValueHeads: c.LinearValueHeads, dnKeyDim: c.LinearKeyDim, dnValueDim: c.LinearValueDim, dnConv: c.LinearConv}
	for i, t := range c.LayerTypes {
		switch t {
		case "linear_attention":
			h.linear[i] = true
		case "full_attention":
		default:
			return modelConfig{}, fmt.Errorf("qwen3: unsupported layer type %q", t)
		}
	}
	if len(c.LayerTypes) != c.Layers || !c.AttnOutputGate || c.Experts <= 0 || c.ExpertsPerToken <= 0 ||
		c.ExpertsPerToken > c.Experts || c.ExpertWidth <= 0 || c.SharedExpertWidth != c.ExpertWidth ||
		c.LinearKeyHeads <= 0 || c.LinearValueHeads%c.LinearKeyHeads != 0 || c.LinearKeyDim <= 0 ||
		c.LinearValueDim <= 0 || c.LinearConv < 1 || h.rotary <= 0 || h.rotary%2 != 0 || h.rotary > c.HeadDim {
		return modelConfig{}, errors.New("qwen3: unsupported Qwen3.5 layout (layer types, gating, experts, DeltaNet, or RoPE)")
	}
	return h, nil
}

// newHead allocates the language-model head's packed storage, which
// loadRows then fills a row range at a time.
func newHead(c *modelConfig, format string, rot *rotation) (linear, error) {
	if format == WeightsInt8 {
		w, err := q8gemm.NewWeightsI8(c.hidden, c.vocab)
		return linear{i8: w, rot: rot}, err
	}
	w, err := q8gemm.NewWeightsF16(c.hidden, c.vocab)
	return linear{f16: w}, err
}

// loadRows reads rows [r0, r1) of an [n][k] BF16 matrix and packs them into
// dst, which newHead allocated.
func loadRows(st *safetensors.Checkpoint, name string, n, k, r0, r1 int, buf []uint16, dst *linear) error {
	t, err := st.Lookup(name, n, k)
	if err != nil {
		return fmt.Errorf("qwen3: %w", err)
	}
	if t.DType != "BF16" {
		return fmt.Errorf("qwen3: %s is %s; the loader expects the official BF16 checkpoint", name, t.DType)
	}
	raw := buf[:(r1-r0)*k]
	if err := t.ReadBits(raw, int64(r0)*int64(k)); err != nil {
		return fmt.Errorf("qwen3: %w", err)
	}
	if dst.i8 == nil {
		if _, err := dst.f16.PackBF16Rows(raw, r0); err != nil {
			return fmt.Errorf("qwen3: pack %s: %w", name, err)
		}
		return nil
	}
	q, scales := quantizeRotatedRows(raw, r1-r0, k, dst.rot)
	if err := dst.i8.PackRows(q, scales, r0); err != nil {
		return fmt.Errorf("qwen3: pack %s: %w", name, err)
	}
	return nil
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
