// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"math"

	"github.com/GetStream/gophonic/internal/q8gemm"
)

type opKind uint8

const (
	opAddNorm opKind = iota
	opRowScale
	opPack
	opProject
	opQKRope
	opAttention
	opAttentionGEMM
	opSwiGLU
)

const (
	packChunkCols   = 256
	swigluChunkCols = 1024
)

func packChunks(cols int) int   { return (cols + packChunkCols - 1) / packChunkCols }
func swigluChunks(cols int) int { return (cols + swigluChunkCols - 1) / swigluChunkCols }

// layerOp is the one reusable parallel operation of a PrefillWorkspace. The
// caller sets kind and the fields that kind reads, then runs it over disjoint
// item ranges. Every item writes only its own rows, columns, or panels.
type layerOp struct {
	ws         *Workspace
	kind       opKind
	rows       int
	layer      *modelLayer
	layerIndex int

	residual, normWeight []float32 // opAddNorm

	src      []float32 // opPack
	cols     int
	panels   int // opProject
	proj     [3]projection
	panelEnd [3]int
}

// ApplyRows implements rangeOp.
func (o *layerOp) ApplyRows(worker, start, end int) {
	switch o.kind {
	case opAddNorm:
		o.addNorm(start, end)
	case opRowScale:
		o.rowScale(start, end)
	case opPack:
		o.pack(start, end)
	case opProject:
		o.project(worker, start, end)
	case opQKRope:
		o.qkRope(start, end)
	case opAttention:
		o.attention(start, end)
	case opAttentionGEMM:
		o.attentionGEMM(worker, start, end)
	case opSwiGLU:
		o.swiglu(start, end)
	}
}

func (o *layerOp) addNorm(start, end int) {
	f := &o.ws.owner.m.cfg
	width := f.hidden
	for r := start; r < end; r++ {
		h := o.ws.h[r*width : (r+1)*width]
		if o.residual != nil {
			addInto(h, o.residual[r*width:(r+1)*width])
		}
		rmsNorm32(o.ws.norm[r*width:(r+1)*width], h, o.normWeight, f.eps)
	}
}

func (o *layerOp) rowScale(start, end int) {
	for r := start; r < end; r++ {
		tile := o.ws.tiles[r/q8gemm.ActivationRows]
		tile.SetRowScale(r%q8gemm.ActivationRows, q8gemm.MaxAbs(o.src[r*o.cols:(r+1)*o.cols]))
	}
}

func (o *layerOp) pack(start, end int) {
	chunks := packChunks(o.cols)
	for item := start; item < end; item++ {
		tile, chunk := item/chunks, item%chunks
		src := o.src[tile*q8gemm.ActivationRows*o.cols:]
		k0 := chunk * packChunkCols
		if err := o.ws.tiles[tile].PackRange(src, o.cols, k0, min(o.cols, k0+packChunkCols)); err != nil {
			panic("qwen3: activation pack: " + err.Error())
		}
	}
}

// project handles whole panels: each item computes one 64-column panel for
// every 16-row tile, so the panel's weights (at most a few MiB) stay in the
// core's L2 cache across tiles instead of streaming from DRAM once per tile.
func (o *layerOp) project(worker, start, end int) {
	tiles := (o.rows + q8gemm.ActivationRows - 1) / q8gemm.ActivationRows
	for panel := start; panel < end; {
		m := 0
		for panel >= o.panelEnd[m] {
			m++
		}
		first := 0
		if m > 0 {
			first = o.panelEnd[m-1]
		}
		// Stop at the end of this projection or of the range.
		stop := min(o.panelEnd[m], end)
		p := o.proj[m]
		_, n := p.w.Dims()
		for tile := range tiles {
			dst := p.dst[tile*q8gemm.ActivationRows*n:]
			if err := q8gemm.MulPanels(dst, n, o.ws.tiles[tile], p.w, panel-first, stop-first, o.ws.scratch[worker]); err != nil {
				panic("qwen3: packed projection: " + err.Error())
			}
		}
		panel = stop
	}
}

func (o *layerOp) qkRope(start, end int) {
	ws, f := o.ws, &o.ws.owner.m.cfg
	hd, half := f.headDim, f.headDim/2
	for r := start; r < end; r++ {
		base := int(ws.rowPos[r]) * half
		cos, sin := ws.ropeCos[base:base+half], ws.ropeSin[base:base+half]
		for head := range f.heads {
			v := ws.q[r*f.hidden+head*hd : r*f.hidden+(head+1)*hd]
			rmsNorm32(v, v, o.layer.qNorm, f.eps)
			rotateHalves(v[:half], v[half:], cos, sin)
		}
		for head := range f.kvHeads {
			v := ws.keys[r*f.kvDim+head*hd : r*f.kvDim+(head+1)*hd]
			rmsNorm32(v, v, o.layer.kNorm, f.eps)
			rotateHalves(v[:half], v[half:], cos, sin)
		}
	}
}

// rmsNorm32 writes RMSNorm(src)*weight into dst (which may alias src).
func rmsNorm32(dst, src, weight []float32, eps float64) {
	mean := float64(sumSquares(src)) / float64(len(src))
	scaleMulInto(dst, src, weight, float32(1/math.Sqrt(mean+eps)))
}

// attention handles one (row, KV head) pair per item: every query head in
// that group attends causally to its own sequence with a streaming softmax,
// so no per-item score scratch is needed.
func (o *layerOp) attention(start, end int) {
	ws, f := o.ws, &o.ws.owner.m.cfg
	hd, group := f.headDim, f.heads/f.kvHeads
	scale := float32(f.attnScale)
	for item := start; item < end; item++ {
		r, g := item/f.kvHeads, item%f.kvHeads
		if ws.rowLen[r] >= gemmAttentionMin {
			continue // handled by attentionGEMM
		}
		first := int(ws.rowStart[r])
		for qh := g * group; qh < (g+1)*group; qh++ {
			q := ws.q[r*f.hidden+qh*hd : r*f.hidden+(qh+1)*hd]
			out := ws.ctx[r*f.hidden+qh*hd : r*f.hidden+(qh+1)*hd]
			clear(out)
			maxScore := float32(math.Inf(-1))
			var sum float32
			for j := first; j <= r; j++ {
				off := j*f.kvDim + g*hd
				s := dot32(q, ws.keys[off:off+hd]) * scale
				if s > maxScore {
					if sum != 0 {
						c := expNonPositive32(maxScore - s)
						sum *= c
						scaleVector(out, c)
					}
					maxScore = s
				}
				p := expNonPositive32(s - maxScore)
				sum += p
				axpy32(out, ws.values[off:off+hd], p)
			}
			scaleVector(out, 1/sum)
		}
	}
}

// attentionGEMM handles blocked causal attention for long sequences. Each item
// packs its KV group's keys and values up to the block's last query once, then
// for every query head in the group computes scores = Q·Kᵀ and ctx = P·V as
// matrix products, never touching keys beyond the causal limit.
func (o *layerOp) attentionGEMM(worker, start, end int) {
	ws, c := o.ws, &o.ws.owner.m.cfg
	hd, group, qdim := c.headDim, c.heads/c.kvHeads, c.heads*c.headDim
	sc := &ws.attnScratch[worker]
	scale := float32(c.attnScale)
	for i := start; i < end; i++ {
		it := ws.attnItems[i]
		base, q0, q1, g, past := int(it.start), int(it.q0), int(it.q1), int(it.group), int(it.past)
		nk, qb := past+q1, q1-q0
		keys, values := ws.keys[base*c.kvDim+g*hd:], ws.values[base*c.kvDim+g*hd:]
		if kv := ws.prefix; kv != nil {
			keys, values = kv.keys[o.layerIndex][g*hd:], kv.values[o.layerIndex][g*hd:]
		}
		must(sc.keysT.Reshape(hd, nk))
		must(sc.keysT.Pack(keys, c.kvDim, true))
		must(sc.values.Reshape(nk, hd))
		must(sc.values.Pack(values, c.kvDim, false))
		scores := sc.scores[:qb*nk]
		for qh := g * group; qh < (g+1)*group; qh++ {
			off := (base+q0)*qdim + qh*hd
			must(sc.keysT.MulScratch(scores, nk, ws.q[off:], qdim, qb, sc.gemm))
			for r := range qb {
				row := scores[r*nk : (r+1)*nk]
				valid := past + q0 + r + 1
				softmaxScaled(row[:valid], scale)
				clear(row[valid:])
			}
			must(sc.values.MulScratch(ws.ctx[off:], qdim, scores, nk, qb, sc.gemm))
		}
	}
}

func must(err error) {
	if err != nil {
		panic("qwen3: blocked attention: " + err.Error())
	}
}

func (o *layerOp) swiglu(start, end int) {
	ws, n := o.ws, o.ws.owner.m.cfg.intermediate
	chunks := swigluChunks(n)
	for item := start; item < end; item++ {
		r, c := item/chunks, item%chunks
		lo := r*n + c*swigluChunkCols
		hi := r*n + min(n, (c+1)*swigluChunkCols)
		gate, up := ws.gate[lo:hi], ws.up[lo:hi]
		swigluInto(gate, up)
	}
}

// silu32 evaluates x*sigmoid(x) with both exponentials on nonpositive inputs.
func silu32(x float32) float32 {
	if x >= 0 {
		return x / (1 + expNonPositive32(-x))
	}
	e := expNonPositive32(x)
	return x * e / (1 + e)
}

// expNonPositive32 evaluates exp(x) for x <= 0 with range reduction onto
// (-ln 2, 0] and a degree-8 Taylor polynomial; relative error is below 1e-6.
// Inputs below -87 return exp(-87).
const (
	ln2Hi = 0.693359375
	ln2Lo = -2.12194440e-4
)

func expNonPositive32(x float32) float32 {
	x = max(x, -87) // exp(-87) is near FP32's smallest normal

	n := int(x * 1.4426950408889634)
	// Cody–Waite reduction: n*ln2Hi is exact (ln2Hi has 11 significant
	// bits), so r is accurate whether or not the compiler fuses into FMA.
	fn := float32(n)
	r := (x - fn*ln2Hi) - fn*ln2Lo
	p := float32(1.0 / 40320.0)
	p = 1.0/5040.0 + r*p
	p = 1.0/720.0 + r*p
	p = 1.0/120.0 + r*p
	p = 1.0/24.0 + r*p
	p = 1.0/6.0 + r*p
	p = 0.5 + r*p
	p = 1 + r*p
	p = 1 + r*p
	return math.Float32frombits(uint32(n+127)<<23) * p
}
