// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
	"github.com/GetStream/gophonic/internal/q8gemm"
)

//go:embed gpu.metal
var gpuSource string

const (
	gpuThreads = 256
	gpuAlign   = 256
)

// gpuLayer holds one layer's projections in one shared buffer: int8 rows
// followed by FP32 row scales, at byte offsets.
type gpuLayer struct {
	buf                                              *metal.Buffer
	qkv, qkvScale, o, oScale, gu, guScale, d, dScale int
}

// gpuModel is a Qwen3 model resident in GPU-visible memory. The residual
// stream is rotated by hidden (R); q, k, v, gate, and up store W·diag(norm)·Rᵀ,
// o and down store R·W, the value rows and o inputs carry a per-head
// Hadamard P, and down's inputs are rotated online by inter.
type gpuModel struct {
	dev                          *metal.Device
	qkv, o, gateup, down, attend *metal.Pipeline
	rotate                       *metal.Pipeline
	layers                       []gpuLayer
	norms, signs, rope           *metal.Buffer
	hidden, inter, head          *rotation
	cfg                          *modelConfig
	bits                         int // 8 or 4
}

// gpuRows is the number of weight rows per GEMV threadgroup: 8 simdgroups
// times rowsPerSimdgroup in gpu.metal.
func gpuRows(bits int) int {
	if bits == 8 {
		return 16
	}
	return 32
}

func alignUp(n int) int { return (n + gpuAlign - 1) &^ (gpuAlign - 1) }

// loadGPU quantizes every projection into GPU buffers.
func (m *Weights) loadGPU(st *safetensors, bits int) error {
	c := &m.cfg
	if c.hidden != 4096 || c.heads != 32 || c.kvHeads != 8 || c.headDim != 128 || c.intermediate%maxRotationBlock != 0 {
		return errors.New("qwen3: the GPU backend supports the Qwen3-8B geometry only")
	}
	dev, err := metal.Open()
	if err != nil {
		return err
	}
	g := &gpuModel{dev: dev, cfg: c, layers: make([]gpuLayer, c.layers), bits: bits}
	suffix := ""
	if bits == 4 {
		suffix = "_q4"
	}
	lib, err := dev.Compile(gpuSource)
	if err != nil {
		return err
	}
	for _, p := range []struct {
		dst  **metal.Pipeline
		name string
	}{{&g.qkv, "gemv_qkv" + suffix}, {&g.o, "gemv_o" + suffix}, {&g.gateup, "gemv_gateup" + suffix}, {&g.down, "gemv_down" + suffix}, {&g.attend, "attend1"}, {&g.rotate, "rotate4096"}} {
		if *p.dst, err = dev.Pipeline(lib, p.name); err != nil {
			return err
		}
	}
	h, kv, inter, qdim := c.hidden, c.kvDim, c.intermediate, c.heads*c.headDim
	g.hidden, g.inter = newRotation(h), newRotation(inter)
	head := newRotation(c.headDim)
	g.head = &rotation{signs: make([]float32, qdim), block: c.headDim, scale: head.scale}
	for i := range g.head.signs {
		g.head.signs[i] = head.signs[i%c.headDim]
	}
	// Down's GPU input rotation takes the 1/sqrt(block) factor with the signs.
	if g.signs, err = dev.Buffer(4 * inter); err != nil {
		return err
	}
	for i, s := range g.inter.signs {
		floats(g.signs.Bytes())[i] = s * g.inter.scale // exact: a power of two
	}
	if g.norms, err = dev.Buffer(4 * 2 * c.headDim * c.layers); err != nil {
		return err
	}
	norms := floats(g.norms.Bytes())
	for i := range m.layers {
		copy(norms[2*i*c.headDim:], m.layers[i].qNorm)
		copy(norms[(2*i+1)*c.headDim:], m.layers[i].kNorm)
	}
	half := c.headDim / 2
	if g.rope, err = dev.Buffer(4 * 2 * maxTokens * half); err != nil {
		return err
	}
	rope := floats(g.rope.Bytes())
	for pos := range maxTokens {
		for d, inv := range c.invFreq {
			theta := float64(pos) * inv
			rope[pos*half+d] = float32(math.Cos(theta))
			rope[maxTokens*half+pos*half+d] = float32(math.Sin(theta))
		}
	}

	type job struct {
		name     string
		n, k     int
		norm     []float32 // folded input RMSNorm weight
		in, out  *rotation // input- and output-side rotations
		layer    *gpuLayer
		base, sc int // byte offsets of row 0 and scale 0
		step     int // destination row stride in rows
		scaleMul float32
		row0     int // destination row of source row 0
	}
	var jobs []job
	for i := range m.layers {
		l, gl := &m.layers[i], &g.layers[i]
		off := 0
		place := func(rows, k int) (int, int) {
			w := off
			off = alignUp(off + rows*k*bits/8)
			s := off
			if bits == 4 {
				off = alignUp(off + 2*rows*(k/q4Group))
			} else {
				off = alignUp(off + 4*rows)
			}
			return w, s
		}
		gl.qkv, gl.qkvScale = place(qdim+2*kv, h)
		gl.o, gl.oScale = place(h, qdim)
		gl.gu, gl.guScale = place(2*inter, h)
		gl.d, gl.dScale = place(h, inter)
		if gl.buf, err = dev.Buffer(off); err != nil {
			return err
		}
		p := fmt.Sprintf("model.layers.%d.", i)
		jobs = append(jobs,
			job{p + "self_attn.q_proj.weight", qdim, h, l.attnNorm, g.hidden, nil, gl, gl.qkv, gl.qkvScale, 1, 1, 0},
			job{p + "self_attn.k_proj.weight", kv, h, l.attnNorm, g.hidden, nil, gl, gl.qkv, gl.qkvScale, 1, 1, qdim},
			job{p + "self_attn.v_proj.weight", kv, h, l.attnNorm, g.hidden, g.head, gl, gl.qkv, gl.qkvScale, 1, 1, qdim + kv},
			job{p + "self_attn.o_proj.weight", h, qdim, nil, g.head, g.hidden, gl, gl.o, gl.oScale, 1, 1, 0},
			job{p + "mlp.gate_proj.weight", inter, h, l.mlpNorm, g.hidden, nil, gl, gl.gu, gl.guScale, 2, 1, 0},
			job{p + "mlp.up_proj.weight", inter, h, l.mlpNorm, g.hidden, nil, gl, gl.gu, gl.guScale, 2, 1, 1},
			job{p + "mlp.down_proj.weight", h, inter, nil, g.inter, g.hidden, gl, gl.d, gl.dScale, 1, 1, 0},
		)
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
			var raw []uint16
			var mat []float32
			for {
				mu.Lock()
				if first != nil || next == len(jobs) {
					mu.Unlock()
					return
				}
				j := jobs[next]
				next++
				mu.Unlock()
				t, err := st.lookup(j.name, j.n, j.k)
				if err == nil && t.dtype != "BF16" {
					err = fmt.Errorf("qwen3: %s is %s; the loader expects the official BF16 checkpoint", j.name, t.dtype)
				}
				if err == nil {
					raw = grow(raw, j.n*j.k)
					err = readInto(t, raw)
				}
				if err != nil {
					mu.Lock()
					first = errors.Join(first, err)
					mu.Unlock()
					return
				}
				mat = grow(mat, j.n*j.k)
				for i, b := range raw {
					mat[i] = q8gemm.BF16ToF32(b)
				}
				for r := range j.n {
					row := mat[r*j.k : (r+1)*j.k]
					if j.norm != nil {
						for i := range row {
							row[i] *= j.norm[i]
						}
					}
					j.in.apply(row)
				}
				if j.out != nil {
					j.out.applyRows(mat, j.n, j.k)
				}
				buf := j.layer.buf.Bytes()
				for r := range j.n {
					row := mat[r*j.k : (r+1)*j.k]
					dst := j.row0 + r*j.step
					if bits == 4 {
						groups := j.k / q4Group
						sc := unsafe.Slice((*uint16)(unsafe.Pointer(&buf[j.sc+2*dst*groups])), groups)
						quantizeRowQ4(row, buf[j.base+dst*j.k/2:][:j.k/2], sc, j.scaleMul)
						continue
					}
					q := unsafe.Slice((*int8)(unsafe.Pointer(&buf[j.base+dst*j.k])), j.k)
					floats(buf[j.sc:])[dst] = quantizeRow(row, q) * j.scaleMul
				}
			}
		}()
	}
	wg.Wait()
	if first != nil {
		return first
	}
	m.gpu = g
	return nil
}

func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}

func floats(b []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/4)
}

// quantizeRow writes round(row/s) to q with s = max|row|/127 and returns s.
func quantizeRow(row []float32, q []int8) float32 {
	m := q8gemm.MaxAbs(row)
	if m == 0 {
		clear(q)
		return 0
	}
	s := m / 127
	for i, v := range row {
		q[i] = int8(max(-127, min(127, math.RoundToEven(float64(v/s)))))
	}
	return s
}

// q4Group is the 4-bit block length along K.
const q4Group = 32

// quantizeRowQ4 stores each 32-value block of row as 4-bit codes plus 8 (value
// j in the low nibble of byte j, value j+16 in the high nibble) with one FP16
// scale. The scale is chosen per block by a small search that minimizes the
// squared rounding error, starting from the Q4_0 choice max/-8.
func quantizeRowQ4(row []float32, dst []byte, scales []uint16, mul float32) {
	for b := range len(row) / q4Group {
		v := row[b*q4Group : (b+1)*q4Group]
		var peak float32
		for _, x := range v {
			if abs32(x) > abs32(peak) {
				peak = x
			}
		}
		best, bestErr := float32(0), float32(math.Inf(1))
		if peak != 0 {
			for step := range 16 {
				d := peak / -8 * (1 - 0.02*float32(step))
				d = f16Bits(q8gemm.F32ToF16(d))
				if d == 0 {
					continue
				}
				var e float32
				for _, x := range v {
					q := max(-8, min(7, float32(math.RoundToEven(float64(x/d)))))
					e += (x - q*d) * (x - q*d)
				}
				if e < bestErr {
					best, bestErr = d, e
				}
			}
		}
		out := dst[b*16 : (b+1)*16]
		for j := range 16 {
			lo, hi := byte(8), byte(8)
			if best != 0 {
				lo = byte(int(max(-8, min(7, math.RoundToEven(float64(v[j]/best))))) + 8)
				hi = byte(int(max(-8, min(7, math.RoundToEven(float64(v[j+16]/best))))) + 8)
			}
			out[j] = lo | hi<<4
		}
		scales[b] = q8gemm.F32ToF16(best * mul)
	}
}

func abs32(x float32) float32 { return math.Float32frombits(math.Float32bits(x) &^ (1 << 31)) }

// gpuWorkspace holds one forward pass's GPU buffers and encoder.
type gpuWorkspace struct {
	g                                       *gpuModel
	h, qkv, ctx, act, kc                    *metal.Buffer
	vc, embedParts                          *metal.Buffer
	attnParts, mlpParts                     *metal.Buffer // residual sums of squares for the next RMSNorm
	enc                                     metal.Encoder
	qkv0Args, qkvArgs, oArgs, guArgs, dArgs gemvArgs
	attn                                    attnArgs
	rows                                    int
}

type gemvArgs struct {
	k, n  uint32
	eps   float32
	parts uint32
}

type attnArgs struct {
	pos, ropeSin uint32
	eps, scale   float32
}

func (g *gpuModel) newWorkspace() (*gpuWorkspace, error) {
	c := g.cfg
	qdim := c.heads * c.headDim
	rows := gpuRows(g.bits)
	w := &gpuWorkspace{g: g, rows: maxTokens}
	var err error
	for _, b := range []struct {
		dst **metal.Buffer
		n   int
	}{
		{&w.h, 4 * maxTokens * c.hidden},
		{&w.qkv, 4 * (qdim + 2*c.kvDim)},
		{&w.ctx, 4 * qdim},
		{&w.act, 4 * c.intermediate},
		{&w.kc, 4 * c.layers * maxTokens * c.kvDim},
		{&w.vc, 4 * c.layers * maxTokens * c.kvDim},
		{&w.embedParts, 4 * maxTokens},
		{&w.attnParts, 4 * c.hidden / rows},
		{&w.mlpParts, 4 * c.hidden / rows},
	} {
		if *b.dst, err = g.dev.Buffer(b.n); err != nil {
			return nil, err
		}
	}
	eps := float32(c.eps)
	parts := uint32(c.hidden / rows)
	w.qkv0Args = gemvArgs{uint32(c.hidden), uint32(qdim + 2*c.kvDim), eps, 1}
	w.qkvArgs = gemvArgs{uint32(c.hidden), uint32(qdim + 2*c.kvDim), eps, parts}
	w.oArgs = gemvArgs{uint32(qdim), uint32(c.hidden), eps, 0}
	w.guArgs = gemvArgs{uint32(c.hidden), uint32(2 * c.intermediate), eps, parts}
	w.dArgs = gemvArgs{uint32(c.intermediate), uint32(c.hidden), eps, 0}
	w.attn = attnArgs{ropeSin: uint32(maxTokens * c.headDim / 2), eps: eps, scale: float32(c.attnScale)}
	return w, nil
}

// gemv encodes one projection: weights at wOff and scales at sOff in buf,
// input x, output y, and the partial sums read (in) and written (out).
func (w *gpuWorkspace) gemv(p *metal.Pipeline, buf *metal.Buffer, wOff, sOff int, x *metal.Buffer, xOff int,
	y *metal.Buffer, yOff int, in, out *metal.Buffer, inOff int, args *gemvArgs) {
	e := &w.enc
	e.SetPipeline(p)
	e.SetBuffer(buf, wOff, 0)
	e.SetBuffer(buf, sOff, 1)
	e.SetBuffer(x, xOff, 2)
	e.SetBuffer(y, yOff, 3)
	e.SetBuffer(in, inOff, 4)
	e.SetBytes(unsafe.Pointer(args), 16, 5)
	e.SetBuffer(out, 0, 6)
	e.Dispatch(metal.Size{X: int(args.n) / gpuRows(w.g.bits), Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
}

// sequence evaluates one sequence token by token and writes its last
// token's post-final-norm state to dst.
func (w *gpuWorkspace) sequence(m *Weights, ids []int, dst []float32) error {
	g, c := w.g, &m.cfg
	if len(ids) > w.rows {
		return fmt.Errorf("qwen3: %d tokens exceeds the GPU context %d", len(ids), w.rows)
	}
	hs := floats(w.h.Bytes())
	embedParts := floats(w.embedParts.Bytes())
	for t, id := range ids {
		row := hs[t*c.hidden : (t+1)*c.hidden]
		m.embedRow(id, row)
		g.hidden.apply(row)
		embedParts[t] = sumSquares(row)
	}
	e := &w.enc
	g.dev.Begin(e, false)
	kvBytes := 4 * maxTokens * c.kvDim
	for t := range ids {
		hOff := 4 * t * c.hidden
		w.attn.pos = uint32(t)
		for i := range g.layers {
			gl := &g.layers[i]
			if i == 0 {
				w.gemv(g.qkv, gl.buf, gl.qkv, gl.qkvScale, w.h, hOff, w.qkv, 0, w.embedParts, w.attnParts, 4*t, &w.qkv0Args)
			} else {
				w.gemv(g.qkv, gl.buf, gl.qkv, gl.qkvScale, w.h, hOff, w.qkv, 0, w.attnParts, w.attnParts, 0, &w.qkvArgs)
			}

			e.SetPipeline(g.attend)
			e.SetBuffer(w.qkv, 0, 0)
			e.SetBuffer(w.kc, i*kvBytes, 1)
			e.SetBuffer(w.vc, i*kvBytes, 2)
			e.SetBuffer(g.norms, 4*2*i*c.headDim, 3)
			e.SetBuffer(g.norms, 4*(2*i+1)*c.headDim, 4)
			e.SetBuffer(g.rope, 0, 5)
			e.SetBuffer(w.ctx, 0, 6)
			e.SetBytes(unsafe.Pointer(&w.attn), 16, 7)
			e.Dispatch(metal.Size{X: c.kvHeads, Y: 1, Z: 1}, metal.Size{X: 32 * c.heads / c.kvHeads, Y: 1, Z: 1})

			w.gemv(g.o, gl.buf, gl.o, gl.oScale, w.ctx, 0, w.h, hOff, w.mlpParts, w.mlpParts, 0, &w.oArgs)
			w.gemv(g.gateup, gl.buf, gl.gu, gl.guScale, w.h, hOff, w.act, 0, w.mlpParts, w.mlpParts, 0, &w.guArgs)

			e.SetPipeline(g.rotate)
			e.SetBuffer(w.act, 0, 0)
			e.SetBuffer(g.signs, 0, 1)
			e.Dispatch(metal.Size{X: c.intermediate / maxRotationBlock, Y: 1, Z: 1}, metal.Size{X: 1024, Y: 1, Z: 1})

			w.gemv(g.down, gl.buf, gl.d, gl.dScale, w.act, 0, w.h, hOff, w.attnParts, w.attnParts, 0, &w.dArgs)
		}
	}
	if err := e.Wait(); err != nil {
		return err
	}
	last := hs[(len(ids)-1)*c.hidden : len(ids)*c.hidden]
	copy(dst, last)
	g.hidden.unapply(dst)
	rmsNorm32(dst, dst, m.finalNorm, c.eps)
	for _, v := range dst {
		if !finite32(v) {
			return errors.New("qwen3: non-finite hidden state")
		}
	}
	return nil
}

func (w *gpuWorkspace) release() {
	for _, b := range []*metal.Buffer{w.h, w.qkv, w.ctx, w.act, w.kc, w.vc, w.embedParts, w.attnParts, w.mlpParts} {
		b.Release()
	}
}

// releaseGPU frees the model's GPU buffers; the Weights are unusable after.
func (m *Weights) releaseGPU() {
	g := m.gpu
	if g == nil {
		return
	}
	for i := range g.layers {
		g.layers[i].buf.Release()
	}
	g.norms.Release()
	g.signs.Release()
	g.rope.Release()
	g.dev.Close()
	m.gpu = nil
}
