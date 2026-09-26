// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
	"github.com/GetStream/gophonic/internal/safetensors"
)

// The Qwen3.5 family on the GPU (Qwen3.6-35B-A3B). Layers either mix tokens
// with a Gated DeltaNet or attend with gated heads; every layer's MLP is a
// mixture of experts whose shared expert is stored as expert E and joins
// each token as one more slot. Projections and experts are gpu.metal's Q8B
// kernels; gpu_hybrid.metal adds the DeltaNet and gated attention.

//go:embed gpu_hybrid.metal
var gpuHybridSource string

// hybridPipelines are the hybrid kernels.
type hybridPipelines struct {
	dnPrep, dnConvState, dnScan, dnNorm *metal.Pipeline
	hqkRope, hattendM, hattend1         *metal.Pipeline
}

func (c *modelConfig) dnQKV() int { return 2*c.dnKeyHeads*c.dnKeyDim + c.dnValueHeads*c.dnValueDim }

// dnIn is a DeltaNet layer's input projection width: q, k, v, z, b, and a.
func (c *modelConfig) dnIn() int { return c.dnQKV() + c.dnValueHeads*c.dnValueDim + 2*c.dnValueHeads }

// dnParams is a DeltaNet layer's parameters in floats (DN_PARAMS).
func (c *modelConfig) dnParams() int {
	return 2*c.dnValueHeads + c.dnValueDim + c.dnQKV()*c.dnConv
}

// dnState is a DeltaNet layer's recurrent state in floats: a DK×DV matrix
// per value head.
func (c *modelConfig) dnState() int { return c.dnValueHeads * c.dnKeyDim * c.dnValueDim }

// dnConvState is a DeltaNet layer's convolution state in floats: its last
// DCONV-1 inputs.
func (c *modelConfig) dnConvState() int { return (c.dnConv - 1) * c.dnQKV() }

func (c *modelConfig) dnLayers() int { return c.layers - c.attnLayers() }

func hybridDefines(c *modelConfig) string {
	return fmt.Sprintf("#define HD %d\n#define HROT %d\n#define DNK %d\n#define DNV %d\n#define DK %d\n#define DV %d\n#define DCONV %d\n",
		c.headDim, c.rotary, c.dnKeyHeads, c.dnValueHeads, c.dnKeyDim, c.dnValueDim, c.dnConv)
}

// hybridGeometry checks that the hybrid kernels fit: heads a multiple of 32
// wide rotating a multiple of 64 dimensions, at most eight query heads per
// key/value head, 128-wide DeltaNet heads, and widths that tile evenly.
func hybridGeometry(c *modelConfig) error {
	qdim := c.heads * c.headDim
	if c.headDim%32 != 0 || c.rotary%64 != 0 || c.heads/c.kvHeads > 8 || c.dnKeyDim != 128 || c.dnValueDim != 128 ||
		c.hidden%mmColumns != 0 || c.intermediate%mmColumns != 0 || c.dnIn()%mmColumns != 0 ||
		(2*qdim+2*c.kvDim)%mmColumns != 0 || c.dnQKV()%256 != 0 || c.allExperts() > 1024 {
		return errors.New("qwen3: the GPU backend does not fit this Qwen3.5 geometry")
	}
	return nil
}

func (g *gpuModel) hybridPipelines(lib *metal.Library) error {
	var err error
	for _, p := range []struct {
		dst  **metal.Pipeline
		name string
	}{{&g.hy.dnPrep, "dn_prep"}, {&g.hy.dnConvState, "dn_conv_state"}, {&g.hy.dnScan, "dn_scan"}, {&g.hy.dnNorm, "dn_norm"},
		{&g.hy.hqkRope, "hqkRope"}, {&g.hy.hattendM, "hattendM"}, {&g.hy.hattend1, "hattend1"}} {
		if *p.dst, err = g.dev.Pipeline(lib, p.name); err != nil {
			return err
		}
	}
	return nil
}

// loadHybrid reads what a Qwen3.5-family decoder keeps on the host: its
// norms, as 1 + w (the family's norms are zero-centered), and each DeltaNet
// layer's decay, time step bias, output norm, and convolution, which
// uploadDeltaNet gives the GPU.
func (m *Weights) loadHybrid(st *safetensors.Checkpoint) error {
	c := &m.cfg
	shift := func(w []float32) []float32 {
		for i := range w {
			w[i]++
		}
		return w
	}
	var err error
	for i := range m.layers {
		l := &m.layers[i]
		p := fmt.Sprintf("%slayers.%d.", m.prefix, i)
		if l.attnNorm, err = st.Float32(p+"input_layernorm.weight", c.hidden); err != nil {
			return err
		}
		if l.mlpNorm, err = st.Float32(p+"post_attention_layernorm.weight", c.hidden); err != nil {
			return err
		}
		shift(l.attnNorm)
		shift(l.mlpNorm)
		if !c.linear[i] {
			if l.qNorm, err = st.Float32(p+"self_attn.q_norm.weight", c.headDim); err != nil {
				return err
			}
			if l.kNorm, err = st.Float32(p+"self_attn.k_norm.weight", c.headDim); err != nil {
				return err
			}
			shift(l.qNorm)
			shift(l.kNorm)
			continue
		}
		a := p + "linear_attn."
		if l.dnNegA, err = st.Float32(a+"A_log", c.dnValueHeads); err != nil {
			return err
		}
		for i, v := range l.dnNegA {
			l.dnNegA[i] = -float32(math.Exp(float64(v)))
		}
		if l.dnDT, err = st.Float32(a+"dt_bias", c.dnValueHeads); err != nil {
			return err
		}
		if l.dnNorm, err = st.Float32(a+"norm.weight", c.dnValueDim); err != nil {
			return err
		}
		if l.dnConv, err = st.Float32(a+"conv1d.weight", c.dnQKV(), 1, c.dnConv); err != nil {
			return err
		}
	}
	if m.finalNorm, err = st.Float32(m.prefix+"norm.weight", c.hidden); err != nil {
		return err
	}
	shift(m.finalNorm)
	return nil
}

// uploadDeltaNet fills the DeltaNet parameter buffer, DN_PARAMS floats per
// DeltaNet layer.
func (g *gpuModel) uploadDeltaNet(m *Weights) error {
	c := g.cfg
	var err error
	if g.dnParams, err = g.dev.Buffer(4 * max(1, c.dnLayers()) * c.dnParams()); err != nil {
		return err
	}
	all := floats(g.dnParams.Bytes())
	for i, gl := range g.layers {
		if !gl.linear {
			continue
		}
		l := &m.layers[i]
		p := all[gl.index*c.dnParams():]
		copy(p, l.dnNegA)
		copy(p[c.dnValueHeads:], l.dnDT)
		copy(p[2*c.dnValueHeads:], l.dnNorm)
		copy(p[2*c.dnValueHeads+c.dnValueDim:], l.dnConv)
	}
	return nil
}

// hybridJobs appends the quantization of layer i: its token mixer and its
// experts, the shared expert as expert E.
func (m *Weights) hybridJobs(jobs []gpuJob, g *gpuModel, i int, gl *gpuLayer, buf []byte, downIn *rotation) []gpuJob {
	c := &m.cfg
	l := &m.layers[i]
	h, kv, qdim, inter, E := c.hidden, c.kvDim, c.heads*c.headDim, c.intermediate, c.experts
	p := fmt.Sprintf("%slayers.%d.", m.prefix, i)
	in := func(name string, n, row0 int) gpuJob {
		return gpuJob{name: name, n: n, k: h, norm: l.attnNorm, in: g.hidden, buf: buf, base: gl.qkv, sc: gl.qkvScale, step: 1, scaleMul: 1, row0: row0}
	}
	out := func(name string, k int) gpuJob {
		return gpuJob{name: name, n: h, k: k, out: g.hidden, buf: buf, base: gl.o, sc: gl.oScale, step: 1, scaleMul: 1}
	}
	if gl.linear {
		a := p + "linear_attn."
		qkv, vd, nv := c.dnQKV(), c.dnValueHeads*c.dnValueDim, c.dnValueHeads
		jobs = append(jobs,
			in(a+"in_proj_qkv.weight", qkv, 0),
			in(a+"in_proj_z.weight", vd, qkv),
			in(a+"in_proj_b.weight", nv, qkv+vd),
			in(a+"in_proj_a.weight", nv, qkv+vd+nv),
			out(a+"out_proj.weight", vd),
		)
	} else {
		a := p + "self_attn."
		jobs = append(jobs,
			in(a+"q_proj.weight", 2*qdim, 0),
			in(a+"k_proj.weight", kv, 2*qdim),
			in(a+"v_proj.weight", kv, 2*qdim+kv),
			out(a+"o_proj.weight", qdim),
		)
	}
	q := p + "mlp."
	router := func(name string, n, row0 int) gpuJob {
		return gpuJob{name: name, n: n, k: h, norm: l.mlpNorm, in: g.hidden, buf: buf, base: gl.router, step: 1, scaleMul: 1, row0: row0, f32: true}
	}
	gateUp := func(name string, row0 int, dims []int, rows [2]int) gpuJob {
		n := inter
		if dims != nil {
			n = dims[0] * dims[1]
		}
		return gpuJob{name: name, n: n, k: h, norm: l.mlpNorm, in: g.hidden, buf: buf, base: gl.gu, sc: gl.guScale, step: 2, scaleMul: 1, row0: row0, rows: rows, dims: dims}
	}
	down := func(name string, row0 int, dims []int, rows [2]int) gpuJob {
		n := h
		if dims != nil {
			n = dims[0] * dims[1]
		}
		return gpuJob{name: name, n: n, k: inter, in: downIn, out: g.hidden, buf: buf, base: gl.d, sc: gl.dScale, step: 1, scaleMul: 1, row0: row0, rows: rows, dims: dims}
	}
	jobs = append(jobs,
		router(q+"gate.weight", E, 0),
		router(q+"shared_expert_gate.weight", 1, E),
		gateUp(q+"shared_expert.gate_proj.weight", E*2*inter, nil, [2]int{}),
		gateUp(q+"shared_expert.up_proj.weight", E*2*inter+1, nil, [2]int{}),
		down(q+"shared_expert.down_proj.weight", E*h, nil, [2]int{}),
	)
	// The routed experts are stacked: gate_up_proj is [E][2W][h], gate rows
	// then up rows, and down_proj [E][h][W]. A chunk per expert and half
	// interleaves gate and up rows; destination row = row0 + source row·step.
	guDims, dDims := []int{E, 2 * inter, h}, []int{E, h, inter}
	for e := range E {
		g0 := e * 2 * inter
		jobs = append(jobs,
			gateUp(q+"experts.gate_up_proj", -g0, guDims, [2]int{g0, g0 + inter}),
			gateUp(q+"experts.gate_up_proj", 1-g0-2*inter, guDims, [2]int{g0 + inter, g0 + 2*inter}),
			down(q+"experts.down_proj", 0, dDims, [2]int{e * h, (e + 1) * h}),
		)
	}
	return jobs
}

// gpuState is a hybrid sequence's recurrent state: each DeltaNet layer's
// memory and its convolution's last inputs, after pos tokens.
type gpuState struct {
	s, conv *metal.Buffer
	pos     int
	used    uint64 // when a snapshot was last taken
	dev     *metal.Device
}

// newStateLike allocates a state of the same size as s.
func newStateLike(s *gpuState) (*gpuState, error) {
	a, err := s.dev.Buffer(len(s.s.Bytes()))
	if err != nil {
		return nil, err
	}
	b, err := s.dev.Buffer(len(s.conv.Bytes()))
	if err != nil {
		a.Release()
		return nil, err
	}
	return &gpuState{s: a, conv: b, dev: s.dev}, nil
}

func (g *gpuModel) newState() (*gpuState, error) {
	c := g.cfg
	s, err := g.dev.Buffer(4 * max(1, c.dnLayers()*c.dnState()))
	if err != nil {
		return nil, err
	}
	conv, err := g.dev.Buffer(4 * max(1, c.dnLayers()*c.dnConvState()))
	if err != nil {
		s.Release()
		return nil, err
	}
	return &gpuState{s: s, conv: conv, dev: g.dev}, nil
}

// reset forgets every token.
func (s *gpuState) reset() {
	clear(s.s.Bytes())
	clear(s.conv.Bytes())
	s.pos = 0
}

// copyFrom makes s src's state.
func (s *gpuState) copyFrom(src *gpuState) {
	copy(s.s.Bytes(), src.s.Bytes())
	copy(s.conv.Bytes(), src.conv.Bytes())
	s.pos = src.pos
}

func (s *gpuState) release() {
	s.s.Release()
	s.conv.Release()
}

// dnArgs is DnArgs in gpu_hybrid.metal.
type dnArgs struct {
	rows   uint32
	eps    float32
	params uint32
}

// hybridWork is a workspace's buffers and arguments for the hybrid layers:
// DeltaNet's convolved q, k, v, its decays and write strengths, and its
// outputs before the gated norm; the state of sequences evaluated without a
// prefix; and each projection's GEMV arguments.
type hybridWork struct {
	dnx, gb, dnOut *metal.Buffer
	own, state     *gpuState
	dn             dnArgs
	// in0 reads the embedding's single partial sum; in and out follow a
	// layer's MoE. [0] is DeltaNet's, [1] attention's.
	in0, in, out [2]gemvArgs
}

func (g *gpuModel) newHybridWork(w *gpuWorkspace) error {
	c := g.cfg
	hw := &w.hy
	var err error
	for _, b := range []struct {
		dst **metal.Buffer
		n   int
	}{
		{&hw.dnx, 4 * gpuPositions * c.dnQKV()},
		{&hw.gb, 8 * gpuPositions * c.dnValueHeads},
		{&hw.dnOut, 4 * gpuPositions * c.dnValueHeads * c.dnValueDim},
	} {
		if *b.dst, err = g.dev.Buffer(b.n); err != nil {
			return err
		}
	}
	if hw.own, err = g.newState(); err != nil {
		return err
	}
	eps := float32(c.eps)
	parts := uint32(c.hidden / gpuRows(g.bits))
	qdim := c.heads * c.headDim
	hw.dn = dnArgs{eps: eps}
	for k, n := range [2][2]int{{c.dnIn(), c.dnValueHeads * c.dnValueDim}, {2*qdim + 2*c.kvDim, qdim}} {
		hw.in0[k] = gemvArgs{uint32(c.hidden), uint32(n[0]), eps, 1}
		hw.in[k] = gemvArgs{uint32(c.hidden), uint32(n[0]), eps, parts}
		hw.out[k] = gemvArgs{uint32(n[1]), uint32(c.hidden), eps, 0}
	}
	return nil
}

func (hw *hybridWork) release() {
	for _, b := range []*metal.Buffer{hw.dnx, hw.gb, hw.dnOut} {
		b.Release()
	}
	if hw.own != nil {
		hw.own.release()
	}
}

// encodeDeltaNet encodes a DeltaNet layer's token mixing for rows rows whose
// input projection is in qkv: the convolution, the scan, and the gated norm
// into ctx, ready for the output projection.
func (w *gpuWorkspace) encodeDeltaNet(gl *gpuLayer, rows int) {
	g, c, e, hw := w.g, w.g.cfg, &w.enc, &w.hy
	hw.dn.rows, hw.dn.params = uint32(rows), uint32(gl.index*c.dnParams())
	args, size := unsafe.Pointer(&hw.dn), int(unsafe.Sizeof(hw.dn))
	convOff, sOff := 4*gl.index*c.dnConvState(), 4*gl.index*c.dnState()
	e.SetPipeline(g.hy.dnPrep)
	e.SetBuffer(w.qkv, 0, 0)
	e.SetBuffer(hw.state.conv, convOff, 1)
	e.SetBuffer(g.dnParams, 0, 2)
	e.SetBuffer(hw.dnx, 0, 3)
	e.SetBuffer(hw.gb, 0, 4)
	e.SetBytes(args, size, 5)
	e.Dispatch(metal.Size{X: 2*c.dnKeyHeads + c.dnValueHeads, Y: rows, Z: 1}, metal.Size{X: 32, Y: 1, Z: 1})

	e.SetPipeline(g.hy.dnConvState)
	e.Dispatch(metal.Size{X: c.dnQKV() / 256, Y: 1, Z: 1}, metal.Size{X: 256, Y: 1, Z: 1})

	e.SetPipeline(g.hy.dnScan)
	e.SetBuffer(hw.dnx, 0, 0)
	e.SetBuffer(hw.gb, 0, 1)
	e.SetBuffer(hw.state.s, sOff, 2)
	e.SetBuffer(hw.dnOut, 0, 3)
	e.Dispatch(metal.Size{X: c.dnValueHeads, Y: 1, Z: 1}, metal.Size{X: 4 * c.dnValueDim, Y: 1, Z: 1})

	e.SetPipeline(g.hy.dnNorm)
	e.SetBuffer(hw.dnOut, 0, 0)
	e.SetBuffer(w.qkv, 0, 1)
	e.SetBuffer(g.dnParams, 0, 2)
	e.SetBuffer(w.ctx, 0, 3)
	e.Dispatch(metal.Size{X: c.dnValueHeads, Y: rows, Z: 1}, metal.Size{X: 32, Y: 1, Z: 1})
}

// encodeHybridBatch is encodeBatch for the Qwen3.5 family.
func (w *gpuWorkspace) encodeHybridBatch(rows int) {
	g, c, e := w.g, w.g.cfg, &w.enc
	qdim := c.heads * c.headDim
	w.attn.pos = 0
	for i := range g.layers {
		gl := &g.layers[i]
		in, parts := w.attnParts, c.hidden/16 // the experts' partial sums
		if i == 0 {
			in, parts = w.embedParts, 1
		}
		if gl.linear {
			w.mmDispatch(0, gl.buf, gl.qkv, gl.qkvScale, w.h, w.qkv, in, w.attnParts, c.hidden, c.dnIn(), rows, parts)
			w.encodeDeltaNet(gl, rows)
			w.mmDispatch(1, gl.buf, gl.o, gl.oScale, w.ctx, w.h, w.mlpParts, w.mlpParts, c.dnValueHeads*c.dnValueDim, c.hidden, rows, 0)
		} else {
			w.mmDispatch(0, gl.buf, gl.qkv, gl.qkvScale, w.h, w.qkv, in, w.attnParts, c.hidden, 2*qdim+2*c.kvDim, rows, parts)
			e.SetPipeline(g.hy.hqkRope)
			e.SetBuffer(w.qkv, 0, 0)
			e.SetBuffer(w.curK, gl.index*w.curStride, 1)
			e.SetBuffer(w.curV, gl.index*w.curStride, 2)
			e.SetBuffer(g.norms, 4*2*i*c.headDim, 3)
			e.SetBuffer(g.norms, 4*(2*i+1)*c.headDim, 4)
			e.SetBuffer(g.rope, 0, 5)
			e.SetBuffer(w.info, 0, 6)
			e.SetBytes(unsafe.Pointer(&w.attn), int(unsafe.Sizeof(w.attn)), 7)
			e.Dispatch(metal.Size{X: c.heads + 2*c.kvHeads, Y: rows, Z: 1}, metal.Size{X: 32, Y: 1, Z: 1})

			e.SetPipeline(g.hy.hattendM)
			e.SetBuffer(w.info, 0, 3)
			e.SetBuffer(w.preK, gl.index*w.preStride, 4)
			e.SetBuffer(w.preV, gl.index*w.preStride, 5)
			e.SetBuffer(w.ctx, 0, 6)
			e.Dispatch(metal.Size{X: c.kvHeads, Y: rows, Z: 1}, metal.Size{X: 32 * c.heads / c.kvHeads, Y: 1, Z: 1})
			w.mmDispatch(1, gl.buf, gl.o, gl.oScale, w.ctx, w.h, w.mlpParts, w.mlpParts, qdim, c.hidden, rows, 0)
		}
		w.encodeMoE(gl, rows, 0, c.hidden/mmColumns)
	}
}

// encodeHybridTokens is encodeTokens for the Qwen3.5 family.
func (w *gpuWorkspace) encodeHybridTokens(n int) {
	g, c, e, hw := w.g, w.g.cfg, &w.enc, &w.hy
	for t := range n {
		hOff := 4 * t * c.hidden
		w.attn.pos = uint32(w.past + t)
		for i := range g.layers {
			gl := &g.layers[i]
			k := 1
			if gl.linear {
				k = 0
			}
			if i == 0 {
				w.gemv(g.qkv, gl.buf, gl.qkv, gl.qkvScale, w.h, hOff, w.qkv, 0, w.embedParts, w.attnParts, 4*t, &hw.in0[k])
			} else {
				w.gemv(g.qkv, gl.buf, gl.qkv, gl.qkvScale, w.h, hOff, w.qkv, 0, w.attnParts, w.attnParts, 0, &hw.in[k])
			}
			if gl.linear {
				w.encodeDeltaNet(gl, 1)
			} else {
				e.SetPipeline(g.hy.hattend1)
				e.SetBuffer(w.qkv, 0, 0)
				e.SetBuffer(w.curK, gl.index*w.curStride, 1)
				e.SetBuffer(w.curV, gl.index*w.curStride, 2)
				e.SetBuffer(g.norms, 4*2*i*c.headDim, 3)
				e.SetBuffer(g.norms, 4*(2*i+1)*c.headDim, 4)
				e.SetBuffer(g.rope, 0, 5)
				e.SetBuffer(w.ctx, 0, 6)
				e.SetBytes(unsafe.Pointer(&w.attn), int(unsafe.Sizeof(w.attn)), 7)
				e.Dispatch(metal.Size{X: c.kvHeads, Y: 1, Z: 1}, metal.Size{X: 32 * 2 * c.heads / c.kvHeads, Y: 1, Z: 1})
			}
			w.gemv(g.o, gl.buf, gl.o, gl.oScale, w.ctx, 0, w.h, hOff, w.mlpParts, w.mlpParts, 0, &hw.out[k])
			w.encodeMoE(gl, 1, hOff, c.hidden/gpuRows(g.bits))
		}
	}
}
