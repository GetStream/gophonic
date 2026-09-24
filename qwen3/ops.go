// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"math"
	"sync/atomic"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/whispergemm"
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
	opPackPrefix
	opAttentionPrefix
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
	scaleRows            bool      // opAddNorm also rotates and scales for the next projection

	src      []float32 // opRotate, opRowScale, opPack
	cols     int
	rot      *rotation // input rotation for int8 projections, else nil
	panels   int       // opProject
	proj     [3]projection
	panelEnd [3]int
	// Projection claims: the low 32 bits hold the next panel SME workers take
	// from the front, the high 32 bits the end of the strips NEON workers
	// take from the back (4 strips of 16 columns per panel).
	claims     atomic.Uint64
	smeWorkers int  // participants below this index run SME panels
	coexec     bool // participants from smeWorkers on run NEON strips
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
		o.projectClaims(worker)
	case opQKRope:
		o.qkRope(start, end)
	case opAttention:
		o.attention(start, end)
	case opAttentionGEMM:
		o.attentionGEMM(worker, start, end)
	case opPackPrefix:
		o.packPrefix(start, end)
	case opAttentionPrefix:
		o.attentionPrefix(worker, start, end)
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
		norm := o.ws.norm[r*width : (r+1)*width]
		rmsNorm32(norm, h, o.normWeight, f.eps)
		if o.scaleRows {
			o.scaleRow(r, norm, width)
		}
	}
}

// scaleRow sets row r's quantization scale for the next projection,
// rotating the row into ws.rotated first for int8 weights.
func (o *layerOp) scaleRow(r int, src []float32, cols int) {
	tile, row := r/q8gemm.ActivationRows, r%q8gemm.ActivationRows
	if o.rot == nil {
		o.ws.tiles[tile].SetRowScale(row, q8gemm.MaxAbs(src))
		return
	}
	dst := o.ws.rotated[r*cols : (r+1)*cols]
	copy(dst, src)
	o.rot.apply(dst)
	o.ws.tilesI8[tile].SetRowScale(row, q8gemm.MaxAbs(dst))
}

// rowScale sets each input row's quantization scale; for int8 projections
// it first writes the rotated row to ws.rotated. o.src points at the
// unrotated input while this stage runs.
func (o *layerOp) rowScale(start, end int) {
	for r := start; r < end; r++ {
		o.scaleRow(r, o.src[r*o.cols:(r+1)*o.cols], o.cols)
	}
}

func (o *layerOp) pack(start, end int) {
	chunks := packChunks(o.cols)
	for item := start; item < end; item++ {
		tile, chunk := item/chunks, item%chunks
		src := o.src[tile*q8gemm.ActivationRows*o.cols:]
		k0 := chunk * packChunkCols
		var err error
		if o.rot != nil {
			err = o.ws.tilesI8[tile].PackRange(src, o.cols, k0, min(o.cols, k0+packChunkCols))
		} else {
			err = o.ws.tiles[tile].PackRange(src, o.cols, k0, min(o.cols, k0+packChunkCols))
		}
		if err != nil {
			panic("qwen3: activation pack: " + err.Error())
		}
	}
}

const stripsPerPanel = q8gemm.OutputPanel / q8gemm.StripColumns

// projectClaims is one participant's share of a projection. SME workers
// claim whole 64-column panels from the front; with co-execution, NEON
// workers claim 16-column strips from the back. One compare-and-swap word
// holds both ends, so neither side ever takes columns the other has claimed.
// Each claim covers every 16-row tile, so a panel's weights stay in cache
// across tiles. int8 results are bit-identical on both sides.
func (o *layerOp) projectClaims(worker int) {
	sme := worker < o.smeWorkers
	if !sme && !o.coexec {
		return
	}
	for {
		old := o.claims.Load()
		front, back := uint32(old), uint32(old>>32)
		if sme {
			if int(front) >= o.panels || (front+1)*stripsPerPanel > back {
				return
			}
			if o.claims.CompareAndSwap(old, uint64(back)<<32|uint64(front+1)) {
				o.projectPanel(worker, int(front))
			}
			continue
		}
		if back == 0 || back-1 < front*stripsPerPanel {
			return
		}
		if o.claims.CompareAndSwap(old, uint64(back-1)<<32|uint64(front)) {
			o.projectStrip(int(back - 1))
		}
	}
}

// projection returns the projection holding global panel, and its first panel.
func (o *layerOp) projection(panel int) (projection, int) {
	m := 0
	for panel >= o.panelEnd[m] {
		m++
	}
	first := 0
	if m > 0 {
		first = o.panelEnd[m-1]
	}
	return o.proj[m], first
}

func (o *layerOp) projectPanel(worker, panel int) {
	tiles := (o.rows + q8gemm.ActivationRows - 1) / q8gemm.ActivationRows
	p, first := o.projection(panel)
	_, n := p.l.dims()
	for tile := range tiles {
		dst := p.dst[tile*q8gemm.ActivationRows*n:]
		var err error
		if p.l.i8 != nil {
			err = q8gemm.MulPanelsI8(dst, n, o.ws.tilesI8[tile], p.l.i8, panel-first, panel-first+1)
		} else {
			err = q8gemm.MulPanels(dst, n, o.ws.tiles[tile], p.l.f16, panel-first, panel-first+1, o.ws.scratch[worker])
		}
		if err != nil {
			panic("qwen3: packed projection: " + err.Error())
		}
	}
}

func (o *layerOp) projectStrip(strip int) {
	tiles := (o.rows + q8gemm.ActivationRows - 1) / q8gemm.ActivationRows
	p, first := o.projection(strip / stripsPerPanel)
	local := strip - first*stripsPerPanel
	if local >= p.l.i8.Strips() {
		return // beyond the last column of a partial panel
	}
	_, n := p.l.dims()
	for tile := range tiles {
		if err := q8gemm.MulStripsI8(p.dst[tile*q8gemm.ActivationRows*n:], n, o.ws.tilesI8[tile], p.l.i8, local, local+1); err != nil {
			panic("qwen3: packed projection: " + err.Error())
		}
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

// attentionGEMM handles blocked causal attention for long sequences without
// a stored prefix. Each item packs its KV group's keys and values up to the
// block's last query once, then for every query head in the group computes
// scores = Q·Kᵀ and ctx = P·V as matrix products, never touching keys beyond
// the causal limit.
func (o *layerOp) attentionGEMM(worker, start, end int) {
	ws, c := o.ws, &o.ws.owner.m.cfg
	hd, group, qdim := c.headDim, c.heads/c.kvHeads, c.heads*c.headDim
	sc := &ws.attnScratch[worker]
	scale := float32(c.attnScale)
	for i := start; i < end; i++ {
		it := ws.attnItems[i]
		base, q0, q1, g, past := int(it.start), int(it.q0), int(it.q1), int(it.group), int(it.past)
		nk, qb := past+q1, q1-q0
		keys, values, stride := ws.keys[base*c.kvDim+g*hd:], ws.values[base*c.kvDim+g*hd:], c.kvDim
		must(sc.keysT.Reshape(hd, nk))
		must(sc.keysT.Pack(keys, stride, true))
		must(sc.values.Reshape(nk, hd))
		must(sc.values.Pack(values, stride, false))
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

// packPrefix brings one KV group's packed copy of the stored prefix up to
// ws.past positions for the current layer: it appends newly stored tokens,
// truncates after a branch, and grows storage by doubling (repacking from the
// raw store).
func (o *layerOp) packPrefix(start, end int) {
	ws, c := o.ws, &o.ws.owner.m.cfg
	kv, hd, past := ws.prefix, c.headDim, ws.past
	pk := &kv.packs[o.layerIndex] // prepared by Workspace.preparePrefixPack
	keys, values := kv.keys[o.layerIndex], kv.values[o.layerIndex]
	for g := start; g < end; g++ {
		n := pk.n[g]
		kt := pk.keysT[g]
		if kt == nil || kt.ColumnCapacity() < past {
			capN := max(64, 2*past)
			var err error
			kt, err = whispergemm.NewPackedB(hd, min(capN, kv.capacity))
			must(err)
			pk.keysT[g], n = kt, 0
		}
		n = min(n, past)
		must(kt.Reshape(hd, past))
		var src []float32 // no new columns: PackColumns only re-pads
		if n < past {
			src = keys[n*c.kvDim+g*hd:]
		}
		must(kt.PackColumns(src, c.kvDim, n))
		// Repack value chunks from the first one that changed.
		chunks := (past + prefixChunk - 1) / prefixChunk
		for len(pk.values[g]) < chunks {
			v, err := whispergemm.NewPackedB(prefixChunk, hd)
			must(err)
			pk.values[g] = append(pk.values[g], v)
		}
		for ch := n / prefixChunk; ch < chunks; ch++ {
			rows := min(prefixChunk, past-ch*prefixChunk)
			v := pk.values[g][ch]
			must(v.Reshape(rows, hd))
			must(v.Pack(values[ch*prefixChunk*c.kvDim+g*hd:], c.kvDim, false))
		}
		pk.n[g] = past
	}
}

// attentionPrefix handles one query block and one query head attending to a
// stored prefix (packed once by packPrefix) plus the sequence's own rows. The
// prefix and own scores share one row buffer and one softmax; P·V sums the
// packed value chunks and the own rows.
func (o *layerOp) attentionPrefix(worker, start, end int) {
	ws, c := o.ws, &o.ws.owner.m.cfg
	hd, group, qdim := c.headDim, c.heads/c.kvHeads, c.heads*c.headDim
	sc := &ws.attnScratch[worker]
	pk := &ws.prefix.packs[o.layerIndex]
	scale := float32(c.attnScale)
	for i := start; i < end; i++ {
		it := ws.attnItems[i]
		base, q0, q1, past := int(it.start), int(it.q0), int(it.q1), int(it.past)
		// Per-group items cover every head of the group and pack the group's
		// own rows once; per-head items (see Workspace.attnPerHead) one head.
		h0, h1 := int(it.group)*group, int(it.group+1)*group
		if ws.attnPerHead {
			h0, h1 = int(it.group), int(it.group)+1
		}
		g := h0 / group
		nk, qb := past+q1, q1-q0
		own := base*c.kvDim + g*hd
		must(sc.keysT.Reshape(hd, q1))
		must(sc.keysT.Pack(ws.keys[own:], c.kvDim, true))
		must(sc.values.Reshape(q1, hd))
		must(sc.values.Pack(ws.values[own:], c.kvDim, false))
		scores := sc.scores[:qb*nk]
		for qh := h0; qh < h1; qh++ {
			off := (base+q0)*qdim + qh*hd
			q := ws.q[off:]
			if past > 0 {
				must(pk.keysT[g].MulScratch(scores, nk, q, qdim, qb, sc.gemm))
			}
			must(sc.keysT.MulScratch(scores[past:], nk, q, qdim, qb, sc.gemm))
			for r := range qb {
				row := scores[r*nk : (r+1)*nk]
				valid := past + q0 + r + 1
				softmaxScaled(row[:valid], scale)
				clear(row[valid:])
			}
			// ctx = Σ chunks P[:, chunk]·Vchunk + P[:, past:]·Vown.
			ctx := ws.ctx[off:]
			must(sc.values.MulScratch(ctx, qdim, scores[past:], nk, qb, sc.gemm))
			tmp := sc.tmp[:qb*hd]
			for ch, v := range pk.values[g][:(past+prefixChunk-1)/prefixChunk] {
				must(v.MulScratch(tmp, hd, scores[ch*prefixChunk:], nk, qb, sc.gemm))
				for r := range qb {
					addInto(ctx[r*qdim:r*qdim+hd], tmp[r*hd:(r+1)*hd])
				}
			}
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
