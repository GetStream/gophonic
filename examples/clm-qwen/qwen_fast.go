// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"bytes"
	"errors"
	"fmt"
	"math"

	"github.com/GetStream/gophonic/internal/q8gemv"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// FastEvaluator is a Qwen3 dense transformer forward pass that reads the
// immutable weight bundle loaded by goinfer, but owns the CPU evaluation path.
// It is intentionally scoped to the dense Qwen3 architecture used by CLM's
// Qwen3-8B encoder. A FastWorkspace belongs to one concurrent sequence.
//
// Weight-only int8 matrices use q8gemv directly: each int8 row is dotted with
// the float32 activation and multiplied by that row's scale. This avoids
// materializing a dequantized copy of each projection. F32 uses a local
// matrix-vector loop; other WeightMat kinds use aikit's reusable fallback.
type FastEvaluator struct {
	w   *decoder.Weights
	cfg decoder.Config

	hidden, heads, kvHeads, headDim, kvDim, intermediate int
	eps, attnScale                                       float64
	invFreq                                              []float64
}

// FastWorkspace owns all activations, scores, RoPE tables, and causal KV for
// one sequence. KV grows to the largest sequence seen and is reused on later
// calls. Keep one workspace per concurrent caller; no locking is needed.
type FastWorkspace struct {
	h, norm, q, k, v, ctx, attn, gate, up []float32
	scores                                []float32
	keys, values                          [][]float32
	ropeCos, ropeSin                      []float64
	capacity                              int
	owner                                 *FastEvaluator
	gemm                                  linalg.Workspace
}

// NewFastEvaluator wraps a CPU-loaded Qwen3 model's weights. The caller keeps
// the decoder.Model alive until all evaluations finish; closing it invalidates
// this evaluator if weights are mmap-backed. Only model loading and tokenization
// use goinfer; the forward pass below is local to gophonic.
func NewFastEvaluator(model *decoder.Model) (*FastEvaluator, error) {
	if model == nil {
		return nil, errors.New("clmqwen: nil Qwen3 model")
	}
	return newFastEvaluator(model.Config(), model.Weights())
}

func newFastEvaluator(cfg *decoder.Config, w *decoder.Weights) (*FastEvaluator, error) {
	if cfg == nil || w == nil {
		return nil, errors.New("clmqwen: missing Qwen3 config or weights")
	}
	c := *cfg
	if c.ModelType != "qwen3" {
		return nil, fmt.Errorf("clmqwen: fast evaluator supports qwen3, got %q", c.ModelType)
	}
	if c.HiddenDim <= 0 || c.NumLayers <= 0 || c.NumHeads <= 0 || c.NumKVHeads <= 0 || c.HeadDim <= 0 || c.IntermediateDim <= 0 || c.MaxPositions <= 0 {
		return nil, errors.New("clmqwen: incomplete Qwen3 geometry")
	}
	if c.NumHeads%c.NumKVHeads != 0 || c.HiddenDim != c.NumHeads*c.HeadDim {
		return nil, errors.New("clmqwen: inconsistent Qwen3 grouped-attention geometry")
	}
	if c.RMSNormEps <= 0 || c.RoPEGlobalBase <= 0 {
		return nil, errors.New("clmqwen: invalid Qwen3 normalization or RoPE configuration")
	}
	if hasJSONValue(c.RopeScaling) || hasJSONValue(c.RopeParameters) {
		return nil, errors.New("clmqwen: scaled RoPE is not supported by the fast evaluator")
	}
	if len(w.Layers) != c.NumLayers || len(w.FinalNorm) != c.HiddenDim || w.Embed.Rows() != c.VocabSize || w.Embed.Cols() != c.HiddenDim {
		return nil, errors.New("clmqwen: Qwen3 weight geometry does not match config")
	}
	for i := range w.Layers {
		l := &w.Layers[i]
		if l.QProj.Rows() != c.NumHeads*c.HeadDim || l.QProj.Cols() != c.HiddenDim ||
			l.KProj.Rows() != c.NumKVHeads*c.HeadDim || l.KProj.Cols() != c.HiddenDim ||
			l.VProj.Rows() != c.NumKVHeads*c.HeadDim || l.VProj.Cols() != c.HiddenDim ||
			l.OProj.Rows() != c.HiddenDim || l.OProj.Cols() != c.NumHeads*c.HeadDim ||
			l.GateProj.Rows() != c.IntermediateDim || l.GateProj.Cols() != c.HiddenDim ||
			l.UpProj.Rows() != c.IntermediateDim || l.UpProj.Cols() != c.HiddenDim ||
			l.DownProj.Rows() != c.HiddenDim || l.DownProj.Cols() != c.IntermediateDim ||
			len(l.PreAttnNorm) != c.HiddenDim || len(l.PreMLPNorm) != c.HiddenDim ||
			len(l.QNorm) != c.HeadDim || len(l.KNorm) != c.HeadDim {
			return nil, fmt.Errorf("clmqwen: layer %d has unsupported or inconsistent Qwen3 weights", i)
		}
		for name, mat := range map[string]*linalg.WeightMat{
			"q": &l.QProj, "k": &l.KProj, "v": &l.VProj, "o": &l.OProj,
			"gate": &l.GateProj, "up": &l.UpProj, "down": &l.DownProj,
		} {
			if mat.Kind() == "" {
				return nil, fmt.Errorf("clmqwen: layer %d %s projection has no weight storage", i, name)
			}
		}
	}
	if w.Embed.Kind() == "" {
		return nil, errors.New("clmqwen: embedding table has no weight storage")
	}
	inv := make([]float64, c.HeadDim/2)
	for d := range inv {
		inv[d] = 1 / math.Pow(c.RoPEGlobalBase, float64(2*d)/float64(c.HeadDim))
	}
	return &FastEvaluator{
		w: w, cfg: c, hidden: c.HiddenDim, heads: c.NumHeads, kvHeads: c.NumKVHeads,
		headDim: c.HeadDim, kvDim: c.NumKVHeads * c.HeadDim, intermediate: c.IntermediateDim,
		eps:       c.RMSNormEps,
		attnScale: 1 / math.Sqrt(float64(c.HeadDim)), invFreq: inv,
	}, nil
}

func hasJSONValue(raw []byte) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) != 0 && !bytes.Equal(trimmed, []byte("null"))
}

// NewWorkspace allocates bounded activation scratch. KV rows are allocated on
// first use and grow only when a longer input arrives.
func (f *FastEvaluator) NewWorkspace() *FastWorkspace {
	if f == nil {
		return nil
	}
	return &FastWorkspace{
		h: make([]float32, f.hidden), norm: make([]float32, f.hidden),
		q: make([]float32, f.heads*f.headDim), k: make([]float32, f.kvDim),
		v: make([]float32, f.kvDim), ctx: make([]float32, f.heads*f.headDim),
		attn: make([]float32, f.hidden), gate: make([]float32, f.intermediate),
		up: make([]float32, f.intermediate), keys: make([][]float32, len(f.w.Layers)),
		values: make([][]float32, len(f.w.Layers)), owner: f,
	}
}

// HiddenLastInto evaluates a tokenized sequence and writes its post-final-RMSNorm
// last-token representation into dst. After the workspace has seen this input
// length, this method performs no heap allocations. ids and dst are caller-owned.
func (f *FastEvaluator) HiddenLastInto(ids []int, dst []float32, ws *FastWorkspace) error {
	if f == nil || ws == nil {
		return errors.New("clmqwen: nil evaluator or workspace")
	}
	if ws.owner != f {
		return errors.New("clmqwen: workspace belongs to another evaluator")
	}
	if len(ids) == 0 {
		return errors.New("clmqwen: empty token sequence")
	}
	if len(ids) > f.cfg.MaxPositions {
		return fmt.Errorf("clmqwen: %d tokens exceeds Qwen3 context %d", len(ids), f.cfg.MaxPositions)
	}
	if len(dst) != f.hidden {
		return fmt.Errorf("clmqwen: destination width %d, want %d", len(dst), f.hidden)
	}
	for i, id := range ids {
		if id < 0 || id >= f.cfg.VocabSize {
			return fmt.Errorf("clmqwen: token %d at position %d is outside vocabulary", id, i)
		}
	}
	if err := ws.ensure(f, len(ids)); err != nil {
		return err
	}
	ws.prepareRoPE(f, len(ids))
	for pos, id := range ids {
		if err := f.forwardToken(ws, id, pos); err != nil {
			return fmt.Errorf("clmqwen: token %d: %w", pos, err)
		}
	}
	rmsNormQwen(dst, ws.h, f.w.FinalNorm, f.hidden, f.eps)
	for _, value := range dst {
		if !finite32(value) {
			return errors.New("clmqwen: non-finite hidden state")
		}
	}
	return nil
}

func (ws *FastWorkspace) ensure(f *FastEvaluator, n int) error {
	if ws.capacity >= n {
		return nil
	}
	capN := max(16, ws.capacity)
	for capN < n {
		if capN > f.cfg.MaxPositions/2 {
			capN = n
			break
		}
		capN *= 2
	}
	if capN > f.cfg.MaxPositions {
		capN = f.cfg.MaxPositions
	}
	if f.kvDim <= 0 || capN > int(^uint(0)>>1)/(2*f.kvDim) {
		return errors.New("clmqwen: KV workspace size overflows int")
	}
	for i := range ws.keys {
		ws.keys[i] = make([]float32, capN*f.kvDim)
		ws.values[i] = make([]float32, capN*f.kvDim)
	}
	ws.scores = make([]float32, capN)
	ws.ropeCos = make([]float64, capN*f.headDim/2)
	ws.ropeSin = make([]float64, capN*f.headDim/2)
	ws.capacity = capN
	return nil
}

func (ws *FastWorkspace) prepareRoPE(f *FastEvaluator, n int) {
	half := f.headDim / 2
	for pos := range n {
		base := pos * half
		for d, inv := range f.invFreq {
			theta := float64(pos) * inv
			ws.ropeCos[base+d] = math.Cos(theta)
			ws.ropeSin[base+d] = math.Sin(theta)
		}
	}
}

func (f *FastEvaluator) forwardToken(ws *FastWorkspace, token, pos int) error {
	f.w.Embed.Row(token, ws.h)
	for layer := range f.w.Layers {
		lw := &f.w.Layers[layer]
		rmsNormQwen(ws.norm, ws.h, lw.PreAttnNorm, f.hidden, f.eps)
		if err := projectInto(&ws.gemm, &lw.QProj, ws.norm, ws.q); err != nil {
			return err
		}
		if err := projectInto(&ws.gemm, &lw.KProj, ws.norm, ws.k); err != nil {
			return err
		}
		if err := projectInto(&ws.gemm, &lw.VProj, ws.norm, ws.v); err != nil {
			return err
		}
		for head := range f.heads {
			rmsNormQwen(ws.q[head*f.headDim:(head+1)*f.headDim], ws.q[head*f.headDim:(head+1)*f.headDim], lw.QNorm, f.headDim, f.eps)
		}
		for head := range f.kvHeads {
			rmsNormQwen(ws.k[head*f.headDim:(head+1)*f.headDim], ws.k[head*f.headDim:(head+1)*f.headDim], lw.KNorm, f.headDim, f.eps)
		}
		rotateQwen(ws.q, f.heads, f.headDim, ws.ropeCos[pos*(f.headDim/2):], ws.ropeSin[pos*(f.headDim/2):])
		rotateQwen(ws.k, f.kvHeads, f.headDim, ws.ropeCos[pos*(f.headDim/2):], ws.ropeSin[pos*(f.headDim/2):])
		row := pos * f.kvDim
		copy(ws.keys[layer][row:row+f.kvDim], ws.k)
		copy(ws.values[layer][row:row+f.kvDim], ws.v)
		fastAttention(ws.q, ws.ctx, ws.scores, ws.keys[layer], ws.values[layer], pos+1, f)
		if err := projectInto(&ws.gemm, &lw.OProj, ws.ctx, ws.attn); err != nil {
			return err
		}
		addVector(ws.h, ws.attn)
		rmsNormQwen(ws.norm, ws.h, lw.PreMLPNorm, f.hidden, f.eps)
		if err := projectInto(&ws.gemm, &lw.GateProj, ws.norm, ws.gate); err != nil {
			return err
		}
		if err := projectInto(&ws.gemm, &lw.UpProj, ws.norm, ws.up); err != nil {
			return err
		}
		for i, g := range ws.gate {
			g64 := float64(g)
			ws.gate[i] = float32(g64/(1+math.Exp(-g64))) * ws.up[i]
		}
		if err := projectInto(&ws.gemm, &lw.DownProj, ws.gate, ws.attn); err != nil {
			return err
		}
		addVector(ws.h, ws.attn)
	}
	return nil
}

func projectInto(ws *linalg.Workspace, w *linalg.WeightMat, x, dst []float32) error {
	if q, scales, w8a8, ok := w.Int8(); ok && !w8a8 {
		q8gemv.MulInto(x, q, scales, dst, w.Cols(), w.Rows())
		return nil
	}
	if dense, ok := w.F32(); ok {
		k := w.Cols()
		for row := range w.Rows() {
			weights := dense[row*k : (row+1)*k]
			var sum float32
			for col, value := range weights {
				sum += value * x[col]
			}
			dst[row] = sum
		}
		return nil
	}
	w.MatmulBTInto(ws, x, dst, 1)
	return nil
}

func rmsNormQwen(dst, src, weight []float32, dim int, eps float64) {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+3 < dim; i += 4 {
		x0, x1 := float64(src[i]), float64(src[i+1])
		x2, x3 := float64(src[i+2]), float64(src[i+3])
		s0 += x0 * x0
		s1 += x1 * x1
		s2 += x2 * x2
		s3 += x3 * x3
	}
	var tail float64
	for ; i < dim; i++ {
		x := float64(src[i])
		tail += x * x
	}
	inv := float32(1 / math.Sqrt(((s0+s1)+(s2+s3)+tail)/float64(dim)+eps))
	for i := range dim {
		dst[i] = (src[i] * inv) * weight[i]
	}
}

func rotateQwen(vec []float32, heads, headDim int, cos, sin []float64) {
	half := headDim / 2
	for head := range heads {
		off := head * headDim
		for d := range half {
			c, s := cos[d], sin[d]
			x1, x2 := float64(vec[off+d]), float64(vec[off+half+d])
			vec[off+d] = float32(x1*c - x2*s)
			vec[off+half+d] = float32(x2*c + x1*s)
		}
	}
}

func fastAttention(q, ctx, scores, keys, values []float32, nKeys int, f *FastEvaluator) {
	clear(ctx)
	group := f.heads / f.kvHeads
	for qh := range f.heads {
		kvh := qh / group
		qoff, koff := qh*f.headDim, kvh*f.headDim
		maxScore := math.Inf(-1)
		for pos := range nKeys {
			row := pos * f.kvDim
			var dot float64
			for d := range f.headDim {
				dot += float64(q[qoff+d]) * float64(keys[row+koff+d])
			}
			score := dot * f.attnScale
			scores[pos] = float32(score)
			if score > maxScore {
				maxScore = score
			}
		}
		var sum float64
		for pos := range nKeys {
			e := math.Exp(float64(scores[pos]) - maxScore)
			scores[pos] = float32(e)
			sum += e
		}
		inv := 1 / sum
		out := ctx[qoff : qoff+f.headDim]
		for pos := range nKeys {
			weight := float32(float64(scores[pos]) * inv)
			row := pos * f.kvDim
			voff := row + koff
			for d := range f.headDim {
				out[d] += values[voff+d] * weight
			}
		}
	}
}

func addVector(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}
