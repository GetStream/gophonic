// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"errors"
	"fmt"
	"math"

	"github.com/GetStream/gophonic/internal/q8gemv"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// PrefillEvaluator runs a Qwen3 prompt a layer at a time. All prompt tokens
// pass through each projection together, so quantized projection rows are
// loaded once and reused across the prompt instead of reread once per token.
// It shares immutable weights and geometry with a FastEvaluator.
type PrefillEvaluator struct {
	fast *FastEvaluator
}

// PrefillWorkspace owns reusable prompt activations, one layer's KV rows, and
// matrix scratch. It belongs to one evaluator and one concurrent inference
// lane. Calls are serial by default: this keeps the single-core path free of
// per-call goroutine and heap costs. Allocate one workspace per concurrent call.
type PrefillWorkspace struct {
	h, norm, q, k, v, ctx, attn, gate, up []float32
	weightRow                             []float32
	keys, values                          []float32
	scores                                []float32
	ropeCos, ropeSin                      []float64
	activationQ                           []int8
	activationScales                      []float32
	capacity                              int
	owner                                 *PrefillEvaluator
	gemm                                  linalg.Workspace
}

// NewPrefillEvaluator builds a layer-batched evaluator over a CPU-loaded
// Qwen3 model. The caller must keep model alive until evaluation finishes.
func NewPrefillEvaluator(model *decoder.Model) (*PrefillEvaluator, error) {
	fast, err := NewFastEvaluator(model)
	if err != nil {
		return nil, err
	}
	return newPrefillEvaluator(fast), nil
}

func newPrefillEvaluator(fast *FastEvaluator) *PrefillEvaluator {
	if fast == nil {
		return nil
	}
	return &PrefillEvaluator{fast: fast}
}

// NewWorkspace allocates a reusable workspace. Activation and KV buffers grow
// on the first call for a new maximum sequence length; later calls up to that
// length do not allocate. Matrix parallelism is disabled for this workspace so
// the single-core evaluator stays allocation-free at the public call boundary.
func (p *PrefillEvaluator) NewWorkspace() *PrefillWorkspace {
	if p == nil || p.fast == nil {
		return nil
	}
	ws := &PrefillWorkspace{owner: p}
	ws.gemm.SetThreshold(int(^uint(0) >> 1))
	return ws
}

// HiddenLastInto evaluates ids with causal attention and writes the
// post-final-RMSNorm representation of the last token to dst. For one token it
// uses the existing q8gemv and scalar projection kernels; for multiple tokens
// it applies each projection row across the full prompt before advancing.
// After workspace warmup, the method allocates no heap memory.
func (p *PrefillEvaluator) HiddenLastInto(ids []int, dst []float32, ws *PrefillWorkspace) error {
	if p == nil || p.fast == nil || ws == nil {
		return errors.New("clmqwen: nil prefill evaluator or workspace")
	}
	if ws.owner != p {
		return errors.New("clmqwen: prefill workspace belongs to another evaluator")
	}
	f := p.fast
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
	seq := len(ids)
	for pos, id := range ids {
		f.w.Embed.Row(id, ws.h[pos*f.hidden:(pos+1)*f.hidden])
	}
	ws.prepareRoPE(f, seq)

	for layer := range f.w.Layers {
		lw := &f.w.Layers[layer]
		normRows(ws.norm, ws.h, lw.PreAttnNorm, seq, f.hidden, f.eps)
		if err := projectBatch(ws, &lw.QProj, ws.norm, ws.q, seq, f.hidden); err != nil {
			return fmt.Errorf("clmqwen: layer %d Q projection: %w", layer, err)
		}
		if err := projectBatch(ws, &lw.KProj, ws.norm, ws.k, seq, f.hidden); err != nil {
			return fmt.Errorf("clmqwen: layer %d K projection: %w", layer, err)
		}
		if err := projectBatch(ws, &lw.VProj, ws.norm, ws.v, seq, f.hidden); err != nil {
			return fmt.Errorf("clmqwen: layer %d V projection: %w", layer, err)
		}
		for pos := range seq {
			qrow := ws.q[pos*f.hidden : (pos+1)*f.hidden]
			krow := ws.k[pos*f.kvDim : (pos+1)*f.kvDim]
			for head := range f.heads {
				start := head * f.headDim
				rmsNormQwen(qrow[start:start+f.headDim], qrow[start:start+f.headDim], lw.QNorm, f.headDim, f.eps)
			}
			for head := range f.kvHeads {
				start := head * f.headDim
				rmsNormQwen(krow[start:start+f.headDim], krow[start:start+f.headDim], lw.KNorm, f.headDim, f.eps)
			}
			base := pos * (f.headDim / 2)
			rotateQwen(qrow, f.heads, f.headDim, ws.ropeCos[base:base+f.headDim/2], ws.ropeSin[base:base+f.headDim/2])
			rotateQwen(krow, f.kvHeads, f.headDim, ws.ropeCos[base:base+f.headDim/2], ws.ropeSin[base:base+f.headDim/2])
			row := pos * f.kvDim
			copy(ws.keys[row:row+f.kvDim], krow)
			copy(ws.values[row:row+f.kvDim], ws.v[row:row+f.kvDim])
		}
		for pos := range seq {
			qrow := ws.q[pos*f.hidden : (pos+1)*f.hidden]
			ctxrow := ws.ctx[pos*f.hidden : (pos+1)*f.hidden]
			fastAttention(qrow, ctxrow, ws.scores, ws.keys, ws.values, pos+1, f)
		}
		if err := projectBatch(ws, &lw.OProj, ws.ctx, ws.attn, seq, f.hidden); err != nil {
			return fmt.Errorf("clmqwen: layer %d O projection: %w", layer, err)
		}
		addRows(ws.h, ws.attn, seq, f.hidden)
		normRows(ws.norm, ws.h, lw.PreMLPNorm, seq, f.hidden, f.eps)
		if err := projectBatch(ws, &lw.GateProj, ws.norm, ws.gate, seq, f.hidden); err != nil {
			return fmt.Errorf("clmqwen: layer %d gate projection: %w", layer, err)
		}
		if err := projectBatch(ws, &lw.UpProj, ws.norm, ws.up, seq, f.hidden); err != nil {
			return fmt.Errorf("clmqwen: layer %d up projection: %w", layer, err)
		}
		for i, gate := range ws.gate[:seq*f.intermediate] {
			x := float64(gate)
			ws.gate[i] = float32(x/(1+math.Exp(-x))) * ws.up[i]
		}
		if err := projectBatch(ws, &lw.DownProj, ws.gate, ws.attn, seq, f.intermediate); err != nil {
			return fmt.Errorf("clmqwen: layer %d down projection: %w", layer, err)
		}
		addRows(ws.h, ws.attn, seq, f.hidden)
	}
	last := ws.h[(seq-1)*f.hidden : seq*f.hidden]
	rmsNormQwen(dst, last, f.w.FinalNorm, f.hidden, f.eps)
	for _, value := range dst {
		if !finite32(value) {
			return errors.New("clmqwen: non-finite hidden state")
		}
	}
	return nil
}

func (ws *PrefillWorkspace) ensure(f *FastEvaluator, n int) error {
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
	maxInt := int(^uint(0) >> 1)
	for _, dim := range [...]int{f.hidden, f.kvDim, f.intermediate, f.headDim / 2} {
		if dim <= 0 || capN > maxInt/dim {
			return errors.New("clmqwen: prefill workspace size overflows int")
		}
	}
	ws.h = make([]float32, capN*f.hidden)
	ws.norm = make([]float32, capN*f.hidden)
	ws.q = make([]float32, capN*f.hidden)
	ws.k = make([]float32, capN*f.kvDim)
	ws.v = make([]float32, capN*f.kvDim)
	ws.ctx = make([]float32, capN*f.hidden)
	ws.attn = make([]float32, capN*f.hidden)
	ws.gate = make([]float32, capN*f.intermediate)
	ws.up = make([]float32, capN*f.intermediate)
	ws.weightRow = make([]float32, max(f.hidden, f.intermediate))
	ws.activationQ = make([]int8, capN*max(f.hidden, f.intermediate))
	ws.activationScales = make([]float32, capN)
	ws.keys = make([]float32, capN*f.kvDim)
	ws.values = make([]float32, capN*f.kvDim)
	ws.scores = make([]float32, capN)
	ws.ropeCos = make([]float64, capN*f.headDim/2)
	ws.ropeSin = make([]float64, capN*f.headDim/2)
	ws.capacity = capN
	return nil
}

func (ws *PrefillWorkspace) prepareRoPE(f *FastEvaluator, n int) {
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

func projectBatch(ws *PrefillWorkspace, w *linalg.WeightMat, src, dst []float32, rows, cols int) error {
	if q, scales, w8a8, ok := w.Int8(); ok {
		if w8a8 {
			return projectW8A8Batch(ws, q, scales, src, dst, rows, cols, w.Rows())
		}
		if rows == 1 {
			q8gemv.MulInto(src[:cols], q, scales, dst[:w.Rows()], cols, w.Rows())
			return nil
		}
		// The serial Q8 matrix path widens each weight row once, then reuses it
		// across every prompt row. The workspace threshold is max int, so this
		// avoids linalg's closure-based parallel dispatch and stays allocation-free.
		linalg.MatmulBTQ8Into(&ws.gemm, src, q, scales, dst, rows, cols, w.Rows())
		return nil
	}
	if dense, ok := w.F32(); ok {
		projectF32Batch(src, dense, dst, rows, cols, w.Rows())
		return nil
	}
	if w.IsInt4() {
		projectInt4Batch(ws.weightRow[:cols], src, w, dst, rows, cols)
		return nil
	}
	return errors.New("clmqwen: unsupported Qwen3 projection storage")
}

func projectF32Batch(src, weights, dst []float32, rows, cols, outputs int) {
	for output := range outputs {
		wrow := weights[output*cols : (output+1)*cols]
		for row := range rows {
			xrow := src[row*cols : (row+1)*cols]
			var sum float32
			for k, weight := range wrow {
				sum += weight * xrow[k]
			}
			dst[row*outputs+output] = sum
		}
	}
}

func projectInt4Batch(weightRow, src []float32, w *linalg.WeightMat, dst []float32, rows, cols int) {
	outputs := w.Rows()
	for output := range outputs {
		w.Row(output, weightRow)
		for row := range rows {
			xrow := src[row*cols : (row+1)*cols]
			var sum float32
			for k, weight := range weightRow {
				sum += weight * xrow[k]
			}
			dst[row*outputs+output] = sum
		}
	}
}

func projectW8A8Batch(ws *PrefillWorkspace, q []int8, scales, src, dst []float32, rows, cols, outputs int) error {
	activationQ := ws.activationQ[:rows*cols]
	activationScales := ws.activationScales[:rows]
	linalg.QuantizeActivationsInto(activationQ, activationScales, src, rows, cols)
	for output := range outputs {
		qrow := q[output*cols : (output+1)*cols]
		weightScale := scales[output]
		for row := range rows {
			activationScale := activationScales[row]
			if activationScale == 0 {
				dst[row*outputs+output] = 0
				continue
			}
			arow := activationQ[row*cols : (row+1)*cols]
			var dot int32
			for k, a := range arow {
				dot += int32(a) * int32(qrow[k])
			}
			dst[row*outputs+output] = float32(dot) * activationScale * weightScale
		}
	}
	return nil
}

func normRows(dst, src, weight []float32, rows, dim int, eps float64) {
	for row := range rows {
		start := row * dim
		rmsNormQwen(dst[start:start+dim], src[start:start+dim], weight, dim, eps)
	}
}

func addRows(dst, src []float32, rows, dim int) {
	for i := range rows * dim {
		dst[i] += src[i]
	}
}
