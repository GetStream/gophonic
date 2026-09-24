// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/thesyncim/vibejson"
)

// Weight formats accepted by LoadModel.
const (
	// WeightsF16 stores every BF16 checkpoint weight exactly: each output row
	// is shifted by a power of two into FP16 range. It is the default.
	WeightsF16 = "f16"
	// WeightsInt8 stores each output row as symmetric int8 with one FP32
	// scale, halving weight memory at a measurable accuracy cost.
	WeightsInt8 = "int8"
)

// Model is an immutable Qwen3 dense decoder prepared for last-hidden-state
// inference on the CPU. Projections are packed once for the SME tile kernel;
// the language-model head is not loaded. A Model may be shared by any number
// of evaluators and workspaces.
type Model struct {
	cfg       modelConfig
	embed     []uint16 // BF16 bits, [vocab][hidden]
	finalNorm []float32
	layers    []modelLayer
	format    string
}

type modelLayer struct {
	attnNorm, mlpNorm, qNorm, kNorm []float32
	q, k, v, o, gate, up, down      *q8gemm.Weights
}

type modelConfig struct {
	hidden, layers, heads, kvHeads, headDim, kvDim int
	intermediate, vocab, maxPositions              int
	eps, attnScale                                 float64
	invFreq                                        []float64
}

type hfConfig struct {
	ModelType         string          `json:"model_type"`
	HiddenSize        int             `json:"hidden_size"`
	Layers            int             `json:"num_hidden_layers"`
	Heads             int             `json:"num_attention_heads"`
	KVHeads           int             `json:"num_key_value_heads"`
	HeadDim           int             `json:"head_dim"`
	Intermediate      int             `json:"intermediate_size"`
	Vocab             int             `json:"vocab_size"`
	MaxPositions      int             `json:"max_position_embeddings"`
	RMSNormEps        float64         `json:"rms_norm_eps"`
	RopeTheta         float64         `json:"rope_theta"`
	RopeScaling       any             `json:"rope_scaling"`
	AttentionBias     bool            `json:"attention_bias"`
	HiddenAct         string          `json:"hidden_act"`
	UseSlidingWindow  bool            `json:"use_sliding_window"`
}

// LoadModel reads an official Qwen3 safetensors snapshot directory
// (config.json plus model.safetensors or a sharded index). format is
// WeightsF16 (or empty) or WeightsInt8. Tensors are read and packed in
// parallel; peak memory is the packed model plus one tensor per loader.
func LoadModel(dir, format string) (*Model, error) {
	if format == "" {
		format = WeightsF16
	}
	if format != WeightsF16 && format != WeightsInt8 {
		return nil, fmt.Errorf("clmqwen: unsupported weight format %q", format)
	}
	if !littleEndian() {
		return nil, errors.New("clmqwen: safetensors loading requires a little-endian CPU")
	}
	cfg, err := readConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	st, err := openSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer st.close()

	m := &Model{cfg: cfg, format: format, layers: make([]modelLayer, cfg.layers)}
	h, kv, inter := cfg.hidden, cfg.kvDim, cfg.intermediate
	qdim := cfg.heads * cfg.headDim
	if m.embed, err = st.bf16("model.embed_tokens.weight", cfg.vocab, h); err != nil {
		return nil, err
	}
	if m.finalNorm, err = st.vector("model.norm.weight", h); err != nil {
		return nil, err
	}
	type job struct {
		name string
		n, k int
		dst  **q8gemm.Weights
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
			if *v.dst, err = st.vector(p+v.name, v.n); err != nil {
				return nil, err
			}
		}
		jobs = append(jobs,
			job{p + "self_attn.q_proj.weight", qdim, h, &l.q},
			job{p + "self_attn.k_proj.weight", kv, h, &l.k},
			job{p + "self_attn.v_proj.weight", kv, h, &l.v},
			job{p + "self_attn.o_proj.weight", h, qdim, &l.o},
			job{p + "mlp.gate_proj.weight", inter, h, &l.gate},
			job{p + "mlp.up_proj.weight", inter, h, &l.up},
			job{p + "mlp.down_proj.weight", h, inter, &l.down},
		)
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
				w, err := st.projection(j.name, j.n, j.k, format, buf)
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

// Format reports the projection weight format.
func (m *Model) Format() string { return m.format }

// WeightBytes reports the resident bytes of packed projection weights.
func (m *Model) WeightBytes() int64 {
	var n int64
	for i := range m.layers {
		l := &m.layers[i]
		for _, w := range [...]*q8gemm.Weights{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			n += int64(w.Bytes())
		}
	}
	return n
}

func (m *Model) embedRow(id int, dst []float32) {
	row := m.embed[id*m.cfg.hidden : (id+1)*m.cfg.hidden]
	for i, b := range row {
		dst[i] = q8gemm.BF16ToF32(b)
	}
}

func readConfig(path string) (modelConfig, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return modelConfig{}, fmt.Errorf("clmqwen: read Qwen3 config: %w", err)
	}
	var c hfConfig
	if err := vibejson.Unmarshal(raw, &c); err != nil {
		return modelConfig{}, fmt.Errorf("clmqwen: parse Qwen3 config: %w", err)
	}
	if c.ModelType != "qwen3" {
		return modelConfig{}, fmt.Errorf("clmqwen: expected model_type qwen3, got %q", c.ModelType)
	}
	if c.HiddenAct != "" && c.HiddenAct != "silu" || c.AttentionBias || c.UseSlidingWindow || c.RopeScaling != nil {
		return modelConfig{}, errors.New("clmqwen: unsupported Qwen3 variant (activation, attention bias, sliding window, or scaled RoPE)")
	}
	if c.HeadDim == 0 && c.Heads > 0 {
		c.HeadDim = c.HiddenSize / c.Heads
	}
	const limit = 1 << 20
	for _, v := range []int{c.HiddenSize, c.Layers, c.Heads, c.KVHeads, c.HeadDim, c.Intermediate, c.Vocab, c.MaxPositions} {
		if v <= 0 || v > limit*16 {
			return modelConfig{}, errors.New("clmqwen: incomplete or implausible Qwen3 geometry")
		}
	}
	if c.Heads%c.KVHeads != 0 || c.HeadDim%2 != 0 || c.RMSNormEps <= 0 || c.RopeTheta <= 0 {
		return modelConfig{}, errors.New("clmqwen: inconsistent Qwen3 attention, normalization, or RoPE configuration")
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

type tensorInfo struct {
	file   *os.File
	dtype  string
	shape  []int
	offset int64
	size   int64
}

type safetensors struct {
	files   []*os.File
	tensors map[string]tensorInfo
}

func openSafetensors(dir string) (*safetensors, error) {
	names := []string{"model.safetensors"}
	if raw, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := vibejson.Unmarshal(raw, &index); err != nil {
			return nil, fmt.Errorf("clmqwen: parse safetensors index: %w", err)
		}
		seen := map[string]bool{}
		names = names[:0]
		for _, file := range index.WeightMap {
			if !seen[file] {
				seen[file] = true
				names = append(names, file)
			}
		}
	}
	st := &safetensors{tensors: map[string]tensorInfo{}}
	for _, name := range names {
		if filepath.Base(name) != name {
			st.close()
			return nil, fmt.Errorf("clmqwen: safetensors shard %q is not a plain file name", name)
		}
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			st.close()
			return nil, fmt.Errorf("clmqwen: open weights: %w", err)
		}
		st.files = append(st.files, f)
		if err := st.readHeader(f); err != nil {
			st.close()
			return nil, fmt.Errorf("clmqwen: %s: %w", name, err)
		}
	}
	return st, nil
}

func (st *safetensors) readHeader(f *os.File) error {
	info, err := f.Stat()
	if err != nil {
		return err
	}
	var lenBuf [8]byte
	if _, err := f.ReadAt(lenBuf[:], 0); err != nil {
		return fmt.Errorf("read header length: %w", err)
	}
	n := binary.LittleEndian.Uint64(lenBuf[:])
	if n == 0 || n > 100<<20 || int64(n)+8 > info.Size() {
		return errors.New("invalid safetensors header length")
	}
	header := make([]byte, n)
	if _, err := f.ReadAt(header, 8); err != nil {
		return fmt.Errorf("read header: %w", err)
	}
	// Safetensors pads the header with trailing spaces.
	header = bytes.TrimRight(header, " ")
	// __metadata__ decodes into an empty entry: its fields are unknown here.
	var entries map[string]struct {
		DType   string  `json:"dtype"`
		Shape   []int   `json:"shape"`
		Offsets []int64 `json:"data_offsets"`
	}
	if err := vibejson.Unmarshal(header, &entries); err != nil {
		return fmt.Errorf("parse header: %w", err)
	}
	base := int64(8 + n)
	for name, e := range entries {
		if name == "__metadata__" {
			continue
		}
		if len(e.Offsets) != 2 || e.Offsets[0] < 0 || e.Offsets[1] < e.Offsets[0] || base+e.Offsets[1] > info.Size() {
			return fmt.Errorf("tensor %s has invalid data offsets", name)
		}
		st.tensors[name] = tensorInfo{file: f, dtype: e.DType, shape: e.Shape, offset: base + e.Offsets[0], size: e.Offsets[1] - e.Offsets[0]}
	}
	return nil
}

func (st *safetensors) close() {
	for _, f := range st.files {
		_ = f.Close()
	}
	st.files = nil
}

func (st *safetensors) lookup(name string, shape ...int) (tensorInfo, error) {
	t, ok := st.tensors[name]
	if !ok {
		return t, fmt.Errorf("clmqwen: checkpoint is missing %s", name)
	}
	count := int64(1)
	for _, d := range shape {
		count *= int64(d)
	}
	if len(t.shape) != len(shape) {
		return t, fmt.Errorf("clmqwen: %s has shape %v, want %v", name, t.shape, shape)
	}
	for i := range shape {
		if t.shape[i] != shape[i] {
			return t, fmt.Errorf("clmqwen: %s has shape %v, want %v", name, t.shape, shape)
		}
	}
	width := map[string]int64{"BF16": 2, "F16": 2, "F32": 4}[t.dtype]
	if width == 0 || t.size != count*width {
		return t, fmt.Errorf("clmqwen: %s has dtype %s and %d bytes for shape %v", name, t.dtype, t.size, shape)
	}
	return t, nil
}

// bf16 reads a BF16 tensor's raw bits into new storage.
func (st *safetensors) bf16(name string, shape ...int) ([]uint16, error) {
	t, err := st.lookup(name, shape...)
	if err != nil {
		return nil, err
	}
	if t.dtype != "BF16" {
		return nil, fmt.Errorf("clmqwen: %s is %s; the loader expects the official BF16 checkpoint", name, t.dtype)
	}
	out := make([]uint16, t.size/2)
	return out, readInto(t, out)
}

func readInto(t tensorInfo, dst []uint16) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(dst))), 2*len(dst))
	if _, err := t.file.ReadAt(b, t.offset); err != nil {
		return fmt.Errorf("clmqwen: read tensor: %w", err)
	}
	return nil
}

// vector reads a 1-D BF16, F16, or F32 tensor as FP32.
func (st *safetensors) vector(name string, n int) ([]float32, error) {
	t, err := st.lookup(name, n)
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	switch t.dtype {
	case "F32":
		b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(out))), 4*n)
		if _, err := t.file.ReadAt(b, t.offset); err != nil {
			return nil, fmt.Errorf("clmqwen: read %s: %w", name, err)
		}
	default:
		raw := make([]uint16, n)
		if err := readInto(t, raw); err != nil {
			return nil, err
		}
		for i, b := range raw {
			if t.dtype == "BF16" {
				out[i] = q8gemm.BF16ToF32(b)
			} else {
				out[i] = f16Bits(b)
			}
		}
	}
	for _, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("clmqwen: %s has a non-finite value", name)
		}
	}
	return out, nil
}

// projection reads an [n][k] BF16 matrix into buf and packs it.
func (st *safetensors) projection(name string, n, k int, format string, buf []uint16) (*q8gemm.Weights, error) {
	t, err := st.lookup(name, n, k)
	if err != nil {
		return nil, err
	}
	if t.dtype != "BF16" {
		return nil, fmt.Errorf("clmqwen: %s is %s; the loader expects the official BF16 checkpoint", name, t.dtype)
	}
	raw := buf[:n*k]
	if err := readInto(t, raw); err != nil {
		return nil, err
	}
	if format == WeightsF16 {
		w, err := q8gemm.NewWeightsF16(k, n)
		if err != nil {
			return nil, err
		}
		if _, err := w.PackBF16(raw); err != nil {
			return nil, fmt.Errorf("clmqwen: pack %s: %w", name, err)
		}
		return w, nil
	}
	q, scales := quantizeRowsInt8(raw, n, k)
	w, err := q8gemm.NewWeights(k, n)
	if err != nil {
		return nil, err
	}
	if err := w.Pack(q, scales); err != nil {
		return nil, fmt.Errorf("clmqwen: pack %s: %w", name, err)
	}
	return w, nil
}

// quantizeRowsInt8 maps each row to round(w/s) with s = max|w|/127.
func quantizeRowsInt8(bf16 []uint16, n, k int) ([]int8, []float32) {
	q := make([]int8, n*k)
	scales := make([]float32, n)
	for row := range n {
		src := bf16[row*k : (row+1)*k]
		var m float32
		for _, b := range src {
			m = max(m, float32(math.Abs(float64(q8gemm.BF16ToF32(b)))))
		}
		if m == 0 {
			continue
		}
		scale := m / 127
		scales[row] = scale
		for i, b := range src {
			q[row*k+i] = int8(max(-127, min(127, math.Round(float64(q8gemm.BF16ToF32(b)/scale)))))
		}
	}
	return q, scales
}

func f16Bits(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	exp, mant := int(h>>10&0x1f), float64(h&0x3ff)
	switch exp {
	case 0:
		return sign * float32(math.Ldexp(mant, -24))
	case 31:
		if mant != 0 {
			return float32(math.NaN())
		}
		return sign * float32(math.Inf(1))
	}
	return sign * float32(math.Ldexp(1+mant/1024, exp-15))
}

func littleEndian() bool {
	v := uint16(1)
	return *(*byte)(unsafe.Pointer(&v)) == 1
}
