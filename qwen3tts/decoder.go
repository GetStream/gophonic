// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"math"
	"runtime"
	"sync/atomic"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

// decoder is one lane's streaming codec state: every layer's left context
// and its scratch. decode allocates nothing.
//
// A frame's work is a chain of products too small to fill a machine, on
// weights too large to stay cached, so the decoder runs one executor worker
// per matrix unit (streaming-mode kernels saturate the unit their cluster
// shares) and keeps both busy: a layer over a few rows is split by output
// columns, which spreads its weight traffic; a wide layer by 32-row blocks
// whose SnakeBeta activations, dilated gathers, and residual sums run in
// the same dispatch as its products; and the transformer's layers run in
// lockstep within one dispatch, its participants meeting at spin barriers.
type decoder struct {
	c      *codec
	exec   *whispergemm.Executor
	pos    int // frames decoded
	op     codecOp
	claim  atomic.Int32  // parts claimed by the current dispatch
	arrive atomic.Int32  // barrier arrivals within the current dispatch
	part   []partScratch // one per executor worker

	sum, proj []float32 // quantizer
	pre       []float32 // pre_conv history + input
	x         []float32 // latent row
	h         []float32 // transformer state
	q, k, v   []float32
	att, o    []float32
	gate, up  []float32
	kc, vc    []float32 // [window][heads·headDim]
	lat       []float32 // transformer output

	upIn  [2][]float32 // upsampler inputs
	dw    [2][]float32 // depthwise history + input, rows × latent
	upX   [2][]float32 // depthwise output / residual stream
	ln    []float32
	pw    []float32
	in0   []float32       // inConv history + input
	b0    []float32       // inConv output
	snake [4][]float32    // per block: a row of SnakeBeta history (+ the frame's rows)
	bOut  [4][2][]float32 // per block: the residual stream, alternating between units
	uh    [4][3][]float32 // per unit: conv1's history of act1 rows (+ the frame's rows)
	y, z  []float32       // a unit's conv1 and conv2 outputs
	colU  []float32       // gathered dilated rows of a few-row unit
	outIn []float32       // outConv history
}

// partScratch is one worker's storage within a dispatch.
type partScratch struct {
	sme    []float32 // SME transpose scratch for the widest product
	a1     []float32 // pre-activated input rows of a row block, after their history
	col    []float32 // their gathered dilated rows
	n      []float32 // the transformer's normalized state
	scores []float32 // one head's attention scores
}

// blockRows is the row block of the SME kernel; fewer rows waste its tiles.
const blockRows = 32

// rows of each stage per frame: 1 → 2 → 4 (latent) → 4 (1536) → 32 → 160 → 640 → 1920.
func (c *codec) newDecoder() *decoder {
	d := &decoder{c: c}
	var err error
	if d.exec, err = whispergemm.NewExecutor(codecWorkers(c.threads)); err != nil {
		panic(err) // only for an invalid worker count
	}
	d.op.d = d
	workers := d.exec.Workers()
	// The most rows one part of a row-block dispatch covers.
	partRows := func(rows int) int {
		blocks := (rows + blockRows - 1) / blockRows
		return (blocks + workers - 1) / workers * blockRows
	}
	qd := codecHeads * codecHeadD
	d.sum, d.proj = make([]float32, codebookDim), make([]float32, codecHidden)
	d.pre = make([]float32, (c.preConv.history()+1)*codecHidden)
	d.x = make([]float32, latent)
	d.h = make([]float32, codecHidden)
	d.q, d.k, d.v, d.att = make([]float32, qd), make([]float32, qd), make([]float32, qd), make([]float32, qd)
	d.o = make([]float32, codecHidden)
	d.gate, d.up = make([]float32, codecInter), make([]float32, codecInter)
	d.kc, d.vc = make([]float32, codecLayers*window*qd), make([]float32, codecLayers*window*qd)
	d.lat = make([]float32, latent)
	rows := 1
	for i := range c.up {
		d.upIn[i] = make([]float32, rows*latent)
		rows *= 2
		d.dw[i] = make([]float32, (6+rows)*latent)
		d.upX[i] = make([]float32, rows*latent)
	}
	d.ln, d.pw = make([]float32, rows*latent), make([]float32, rows*4*latent)
	d.in0 = make([]float32, (c.inConv.history()+rows)*latent)
	d.b0 = make([]float32, rows*c.inConv.out)
	kmax, a1, col, colU, y := 0, 0, 0, 0, 0
	for _, l := range []*dense{&c.firstProj, &c.restProj, &c.preConv.w, &c.inProj, &c.outProj, &c.inConv.w, &c.outConv.w} {
		kmax = max(kmax, l.in)
	}
	for i := range c.up {
		kmax = max(kmax, c.up[i].trans.in, c.up[i].pw1.in, c.up[i].pw2.in)
	}
	for i := range c.blocks {
		b := &c.blocks[i]
		d.snake[i] = make([]float32, (1+rows)*b.in)
		a1 = max(a1, (partRows(rows)+1)*b.in)
		kmax = max(kmax, b.trans.w.in)
		rows *= b.rate
		for j := range d.bOut[i] {
			d.bOut[i][j] = make([]float32, rows*b.out)
		}
		for j := range b.units {
			u := &b.units[j]
			h := u.conv1.history()
			d.uh[i][j] = make([]float32, (h+rows)*b.out)
			a1 = max(a1, (partRows(rows)+h)*b.out)
			kmax = max(kmax, u.conv1.w.in, u.conv2.in)
			if u.conv1.dilation > 1 {
				col = max(col, partRows(rows)*u.conv1.k*b.out)
				if rows < 2*blockRows {
					colU = max(colU, rows*u.conv1.k*b.out)
				}
			}
		}
		y = max(y, rows*b.out)
	}
	d.y, d.z, d.colU = make([]float32, y), make([]float32, y), make([]float32, colU)
	oc := &c.outConv
	d.outIn = make([]float32, oc.history()*oc.in)
	a1 = max(a1, (partRows(rows)+oc.history())*oc.in)
	d.part = make([]partScratch, workers)
	for i := range d.part {
		d.part[i] = partScratch{sme: make([]float32, whispergemm.ScratchLen(kmax)), a1: make([]float32, a1), col: make([]float32, col),
			n: make([]float32, codecHidden), scores: make([]float32, window)}
	}
	return d
}

// codecWorkers picks the codec's executor workers: one per matrix unit when
// the kernels run in streaming mode (more only contend for the unit their
// cluster shares), else every performance core.
func codecWorkers(threads int) int {
	if units := matrixUnits(); whispergemm.PackedVectorAccelerated() && units > 0 {
		threads = min(threads, units)
	}
	return max(1, threads)
}

// reset starts a new utterance.
func (d *decoder) reset() {
	d.pos = 0
	clear(d.pre)
	for i := range d.dw {
		clear(d.dw[i])
	}
	clear(d.in0)
	for i := range d.snake {
		clear(d.snake[i])
		for j := range d.uh[i] {
			clear(d.uh[i][j])
		}
	}
	clear(d.outIn)
}

// decode turns one frame's codes into FrameSamples samples of pcm.
func (d *decoder) decode(frame *[groups]int, pcm []float32) {
	c := d.c
	scratch := d.part[0].sme
	// Codebooks: the first and the rest have separate projections.
	copy(d.sum, c.firstTable[frame[0]*codebookDim:(frame[0]+1)*codebookDim])
	c.firstProj.mul(d.proj, d.sum, codebookDim, 1, scratch)
	c.firstProj.bias(d.proj, 1)
	h := c.preConv.history()
	pre := d.pre[h*codecHidden:]
	copy(pre, d.proj)
	clear(d.sum)
	for g := 1; g < groups; g++ {
		addTo(d.sum, c.restTables[((g-1)*codes+frame[g])*codebookDim:][:codebookDim])
	}
	c.restProj.mul(d.proj, d.sum, codebookDim, 1, scratch)
	c.restProj.bias(d.proj, 1)
	addTo(pre[:codecHidden], d.proj)
	d.cols(&c.preConv.w, d.x, d.pre, codecHidden, 1, true, false)
	copy(d.pre[:h*codecHidden], d.pre[codecHidden:(1+h)*codecHidden])
	d.transformer()

	// Two upsamplers: a stride-2 transposed convolution, then ConvNeXt.
	copy(d.upIn[0], d.lat)
	rows := 1
	for i := range c.up {
		u := &c.up[i]
		hist := d.dw[i]
		d.cols(&u.trans, hist[6*latent:], d.upIn[i], latent, rows, true, false)
		rows *= 2
		d.convNeXt(u, hist, rows, d.upX[i])
		if i+1 < len(c.up) {
			copy(d.upIn[i+1], d.upX[i][:rows*latent])
		}
	}
	h = c.inConv.history()
	copy(d.in0[h*latent:], d.upX[1][:rows*latent])
	d.cols(&c.inConv.w, d.b0, d.in0, latent, rows, true, false)
	copy(d.in0[:h*latent], d.in0[rows*latent:(rows+h)*latent])
	x := d.b0
	for i := range c.blocks {
		b := &c.blocks[i]
		hist := d.snake[i]
		out := d.bOut[i][0]
		if rows < 2*blockRows {
			// The transposed convolution reads rows t-1 and t: keep the last.
			b.snake.apply(hist[b.in:], x[:rows*b.in], nil, b.in)
			d.cols(&b.trans.w, out, hist, b.in, rows, true, false)
			copy(hist[:b.in], hist[rows*b.in:(rows+1)*b.in])
		} else {
			d.blocks(&b.snake, &b.trans, hist[:b.in], x, out, rows, nil)
		}
		rows *= b.rate
		for j := range b.units {
			d.unit(b, &b.units[j], d.uh[i][j], d.bOut[i][j&1], d.bOut[i][1-j&1], rows)
		}
		x = d.bOut[i][1]
	}
	d.blocks(&c.outSnake, &c.outConv, d.outIn, x, pcm, rows, nil)
	for i, v := range pcm[:rows] {
		pcm[i] = min(max(v, -1), 1)
	}
	d.pos++
}

// unit runs a residual unit over rows of src into dst: src + conv2(act2(
// conv1(act1(src)))), with hist holding conv1's history of act1 rows.
func (d *decoder) unit(b *block, u *resUnit, hist, src, dst []float32, rows int) {
	ch, h := b.out, u.conv1.history()
	if rows >= 2*blockRows {
		d.blocks(&u.act1, &u.conv1, hist[:h*ch], src, dst, rows, u)
		return
	}
	// A few rows: split the products by columns, the rest here.
	u.act1.apply(hist[h*ch:], src[:rows*ch], nil, ch)
	a, aStride := hist, ch
	if u.conv1.dilation > 1 {
		gather(d.colU, hist, &u.conv1, rows)
		a, aStride = d.colU, u.conv1.k*ch
	}
	y := d.y[:rows*ch]
	d.cols(&u.conv1.w, y, a, aStride, rows, false, false)
	u.act2.apply(y, y, u.conv1.w.b, ch)
	z := d.z[:rows*ch]
	d.cols(&u.conv2, z, y, ch, rows, false, false)
	residual(dst[:rows*ch], src, z, u.conv2.b, ch)
	copy(hist[:h*ch], hist[rows*ch:(rows+h)*ch])
}

// gather writes the k dilated input rows of each of rows outputs, from rows
// that follow their history in hist, one row of k·in per output.
func gather(col, hist []float32, c *conv, rows int) {
	for t := range rows {
		for j := range c.k {
			copy(col[(t*c.k+j)*c.in:(t*c.k+j+1)*c.in], hist[(t+j*c.dilation)*c.in:(t+j*c.dilation+1)*c.in])
		}
	}
}

// convNeXt runs a ConvNeXt block over rows inputs that follow six rows of
// history in hist, writing the result to out.
func (d *decoder) convNeXt(u *upsampler, hist []float32, rows int, out []float32) {
	x := hist[6*latent:]
	// Depthwise causal convolution, kernel 7.
	for t := range rows {
		o := out[t*latent : (t+1)*latent]
		copy(o, u.dwB)
		for j := range 7 {
			in := hist[(t+j)*latent : (t+j+1)*latent]
			for ch := range latent {
				o[ch] += u.dw[ch*7+j] * in[ch]
			}
		}
	}
	// LayerNorm, then the pointwise MLP.
	for t := range rows {
		layerNorm(d.ln[t*latent:(t+1)*latent], out[t*latent:(t+1)*latent], u.normW, u.normB)
	}
	d.cols(&u.pw1, d.pw, d.ln, latent, rows, true, true)
	d.cols(&u.pw2, d.ln, d.pw, 4*latent, rows, true, false)
	for t := range rows {
		o, res := out[t*latent:(t+1)*latent], x[t*latent:(t+1)*latent]
		l := d.ln[t*latent : (t+1)*latent]
		for ch := range latent {
			o[ch] = res[ch] + u.gamma[ch]*l[ch]
		}
	}
	copy(hist[:6*latent], hist[rows*latent:(rows+6)*latent])
}

// transformer runs the eight sliding-window layers on this frame's latent
// row d.x, writing d.lat.
func (d *decoder) transformer() {
	d.cols(&d.c.inProj, d.h, d.x, latent, 1, true, false)
	op := &d.op
	op.kind, op.parts = kindLayers, min(d.exec.Workers(), 2)
	d.claim.Store(0)
	d.arrive.Store(0)
	d.exec.Rows(op, op.parts, 1)
}

// Dispatches. Each covers parts [0, n) of one kind; a part writes only its
// own outputs.
const (
	kindCols   = iota // column chunks of a layer over the same rows
	kindBlocks        // 32-row blocks of a convolution's pre-activated input
	kindLayers        // the transformer's layers, in lockstep
)

// codecOp is the decoder's executor operation.
type codecOp struct {
	d    *decoder
	kind int

	// kindCols: layer over rows of src, aStride apart, into dst, plus the
	// bias when bias is set (a caller may fuse it into a later pass), then
	// GELU when gelu is.
	layer      *dense
	src, dst   []float32
	aStride    int
	rows       int
	bias, gelu bool

	// kindBlocks: dst[t] = conv(pre(src[t-h..t])) over rows of src, with
	// hist the h pre-activated rows before them; with unit set, the rest of
	// the residual unit: dst[t] = src[t] + conv2(act2(conv1(act1(src)))).
	// stage is the new history, in the scratch of the part that ended last.
	pre   *snake
	conv  *conv
	hist  []float32
	unit  *resUnit
	stage []float32

	// kindLayers: parts participants.
	parts int
}

func (d *decoder) cols(l *dense, dst, src []float32, aStride, rows int, bias, gelu bool) {
	op := &d.op
	op.kind, op.layer, op.dst, op.src, op.aStride, op.rows, op.bias, op.gelu = kindCols, l, dst, src, aStride, rows, bias, gelu
	d.claim.Store(0)
	d.exec.Rows(op, len(l.parts), 1)
}

func (d *decoder) blocks(pre *snake, c *conv, hist, src, dst []float32, rows int, u *resUnit) {
	op := &d.op
	op.kind, op.pre, op.conv, op.hist, op.src, op.dst, op.rows, op.unit = kindBlocks, pre, c, hist, src, dst, rows, u
	d.claim.Store(0)
	d.exec.Rows(op, (rows+blockRows-1)/blockRows, 1)
	copy(hist, op.stage)
}

// ApplyRows runs parts [start, end) on one worker.
func (op *codecOp) ApplyRows(start, end int) {
	p := &op.d.part[op.d.claim.Add(1)-1]
	switch op.kind {
	case kindCols:
		l := op.layer
		for c := start; c < end; c++ {
			c0 := c * l.cols
			width := min(l.cols, l.out-c0)
			l.parts[c].MulScratch(op.dst[c0:], l.out, op.src, op.aStride, op.rows, p.sme)
			for r := range op.rows {
				row := op.dst[r*l.out+c0 : r*l.out+c0+width]
				if op.bias && l.b != nil {
					addTo(row, l.b[c0:c0+width])
				}
				if op.gelu {
					for i, v := range row {
						row[i] = gelu(v)
					}
				}
			}
		}
	case kindBlocks:
		op.blocks(p, start, end)
	case kindLayers:
		op.layers(p, start)
	}
}

func (op *codecOp) blocks(p *partScratch, start, end int) {
	c := op.conv
	in, h := c.in, c.history()
	r0, r1 := start*blockRows, min(end*blockRows, op.rows)
	n := r1 - r0
	// The pre-activated rows r0-h .. r1: those before the frame from hist.
	a1 := p.a1[:(n+h)*in]
	from := max(0, h-r0)
	if from > 0 {
		copy(a1[:from*in], op.hist[r0*in:h*in])
	}
	op.pre.apply(a1[from*in:], op.src[(r0-h+from)*in:r1*in], nil, in)
	if r1 == op.rows {
		op.stage = a1[n*in:]
	}
	a, aStride := a1, in
	if c.dilation > 1 {
		gather(p.col, a1, c, n)
		a, aStride = p.col, c.k*in
	}
	u := op.unit
	if u == nil {
		y := op.dst[r0*c.out : r1*c.out]
		c.w.mul(y, a, aStride, n, p.sme)
		c.w.bias(y, n)
		return
	}
	y := op.d.y[r0*c.out : r1*c.out]
	c.w.mul(y, a, aStride, n, p.sme)
	u.act2.apply(y, y, c.w.b, c.out)
	z := op.d.z[r0*c.out : r1*c.out]
	u.conv2.mul(z, y, c.out, n, p.sme)
	residual(op.dst[r0*c.out:r1*c.out], op.src[r0*c.out:], z, u.conv2.b, c.out)
}

// layers runs the transformer as participant me of op.parts: each takes
// every parts-th chunk of a projection and head of the attention, and they
// meet at a barrier wherever one's next step reads the other's outputs.
func (op *codecOp) layers(p *partScratch, me int) {
	d, c := op.d, op.d.c
	parts, phase := op.parts, 0
	qd, half := codecHeads*codecHeadD, codecHeadD/2
	slot := d.pos % window
	span := min(d.pos+1, window)
	cos, sin := c.cos[d.pos*half:(d.pos+1)*half], c.sin[d.pos*half:(d.pos+1)*half]
	scale := float32(1 / math.Sqrt(codecHeadD))
	mine := func(i int) bool { return i%parts == me }
	for i := range c.layers {
		l := &c.layers[i]
		rmsNorm(p.n, d.h, l.inNorm)
		n := 0
		for j, v := range [3]*whispergemm.PackedVector{l.q, l.k, l.v} {
			out := [3][]float32{d.q, d.k, d.v}[j]
			for ch := range v.Chunks() {
				if mine(n) {
					v.MulChunks(out, p.n, ch, ch+1)
				}
				n++
			}
		}
		op.barrier(&phase)
		keys, values := d.layerK(i), d.layerV(i)
		for hd := range codecHeads {
			if !mine(hd) {
				continue
			}
			q, k := d.q[hd*codecHeadD:(hd+1)*codecHeadD], d.k[hd*codecHeadD:(hd+1)*codecHeadD]
			rope(q, cos, sin)
			rope(k, cos, sin)
			copy(keys[slot*qd+hd*codecHeadD:], k)
			copy(values[slot*qd+hd*codecHeadD:], d.v[hd*codecHeadD:(hd+1)*codecHeadD])
			top := float32(math.Inf(-1))
			for j := range span {
				s := (d.pos - j) % window
				k := keys[s*qd+hd*codecHeadD : s*qd+(hd+1)*codecHeadD]
				var dot float32
				for e := range codecHeadD {
					dot += q[e] * k[e]
				}
				p.scores[j] = dot * scale
				top = max(top, p.scores[j])
			}
			var sum float32
			for j := range span {
				p.scores[j] = float32(math.Exp(float64(p.scores[j] - top)))
				sum += p.scores[j]
			}
			o := d.att[hd*codecHeadD : (hd+1)*codecHeadD]
			clear(o)
			for j := range span {
				s := (d.pos - j) % window
				w := p.scores[j] / sum
				v := values[s*qd+hd*codecHeadD : s*qd+(hd+1)*codecHeadD]
				for e := range codecHeadD {
					o[e] += w * v[e]
				}
			}
		}
		op.barrier(&phase)
		for j := range l.o {
			if mine(j) {
				op.half(l.o[j], d.att, j, l.attnScale)
			}
		}
		op.barrier(&phase)
		rmsNorm(p.n, d.h, l.postNorm)
		n = 0
		for j, v := range [2]*whispergemm.PackedVector{l.gate, l.up} {
			out := [2][]float32{d.gate, d.up}[j]
			for ch := range v.Chunks() {
				if mine(n) {
					v.MulChunks(out, p.n, ch, ch+1)
				}
				n++
			}
		}
		// A participant holds the same chunks of gate and up.
		for ch := range l.gate.Chunks() {
			if mine(ch) {
				for e := ch * vecChunk; e < min(codecInter, (ch+1)*vecChunk); e++ {
					d.gate[e] = silu(d.gate[e]) * d.up[e]
				}
			}
		}
		op.barrier(&phase)
		for j := range l.down {
			if mine(j) {
				op.half(l.down[j], d.gate, j, l.mlpScale)
			}
		}
		op.barrier(&phase)
	}
	rmsNorm(p.n, d.h, c.norm)
	out := &c.outProj
	for ch := range out.parts {
		if !mine(ch) {
			continue
		}
		c0 := ch * out.cols
		width := min(out.cols, out.out-c0)
		out.parts[ch].MulScratch(d.lat[c0:], out.out, p.n, codecHidden, 1, p.sme)
		if out.b != nil {
			addTo(d.lat[c0:c0+width], out.b[c0:c0+width])
		}
	}
}

// half adds half j of a projection of x, scaled, to the state.
func (op *codecOp) half(w *whispergemm.PackedVector, x []float32, j int, scale []float32) {
	d := op.d
	e0 := j * codecHidden / 2
	o := d.o[e0 : e0+codecHidden/2]
	w.Mul(o, x)
	for e, v := range o {
		d.h[e0+e] += scale[e0+e] * v
	}
}

// vecChunk is the outputs per PackedVector chunk.
const vecChunk = 512

// barrier waits until every participant has arrived at this phase.
func (op *codecOp) barrier(phase *int) {
	if op.parts == 1 {
		return
	}
	*phase++
	target := int32(*phase * op.parts)
	arrive := &op.d.arrive
	arrive.Add(1)
	for spins := 0; arrive.Load() < target; spins++ {
		if spins&1023 == 1023 {
			runtime.Gosched()
		}
	}
}

// layerK and layerV are layer i's key and value windows.
func (d *decoder) layerK(i int) []float32 {
	n := window * codecHeads * codecHeadD
	return d.kc[i*n : (i+1)*n]
}

func (d *decoder) layerV(i int) []float32 {
	n := window * codecHeads * codecHeadD
	return d.vc[i*n : (i+1)*n]
}

func rope(x, cos, sin []float32) {
	half := len(x) / 2
	for i := range half {
		a, b := x[i], x[i+half]
		x[i] = a*cos[i] - b*sin[i]
		x[i+half] = b*cos[i] + a*sin[i]
	}
}

func rmsNorm(dst, x, w []float32) {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+codecEps))
	for i, v := range x {
		dst[i] = v * inv * w[i]
	}
}

func layerNorm(dst, x, w, b []float32) {
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var variance float64
	for _, v := range x {
		dv := float64(v) - mean
		variance += dv * dv
	}
	inv := 1 / math.Sqrt(variance/float64(len(x))+1e-6)
	for i, v := range x {
		dst[i] = float32((float64(v)-mean)*inv)*w[i] + b[i]
	}
}

func gelu(x float32) float32 {
	return float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
}
