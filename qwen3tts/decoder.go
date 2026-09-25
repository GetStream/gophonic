// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"math"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

// decoder is one lane's streaming codec state: every layer's left context
// and its scratch. decode allocates nothing.
type decoder struct {
	c    *codec
	exec *whispergemm.Executor
	pos  int // frames decoded

	sum, proj []float32 // quantizer
	pre       []float32 // pre_conv history + input
	x         []float32 // latent row
	h, n      []float32 // transformer state and normalized state
	q, k, v   []float32
	att, o    []float32
	gate, up  []float32
	kc, vc    []float32 // [window][heads·headDim]
	scores    []float32
	lat       []float32 // transformer output

	upIn  [2][]float32 // upsampler inputs
	dw    [2][]float32 // depthwise history + input, rows × latent
	upX   [2][]float32 // depthwise output / residual stream
	ln    []float32
	pw    []float32
	in0   []float32 // inConv history + input
	b0    []float32 // inConv output
	snake [4][]float32
	bOut  [4][]float32
	unit  [4][3][]float32 // conv1 history + input per unit
	col   []float32       // gathered dilated rows
	y     []float32
	outIn []float32
}

// rows of each stage per frame: 1 → 2 → 4 (latent) → 4 (1536) → 32 → 160 → 640 → 1920.
func (c *codec) newDecoder() *decoder {
	d := &decoder{c: c}
	var err error
	if d.exec, err = whispergemm.NewExecutor(max(1, c.threads)); err != nil {
		panic(err) // only for an invalid worker count
	}
	qd := codecHeads * codecHeadD
	d.sum, d.proj = make([]float32, codebookDim), make([]float32, codecHidden)
	d.pre = make([]float32, (c.preConv.history()+1)*codecHidden)
	d.x = make([]float32, latent)
	d.h, d.n = make([]float32, codecHidden), make([]float32, codecHidden)
	d.q, d.k, d.v, d.att = make([]float32, qd), make([]float32, qd), make([]float32, qd), make([]float32, qd)
	d.o = make([]float32, codecHidden)
	d.gate, d.up = make([]float32, codecInter), make([]float32, codecInter)
	d.kc, d.vc = make([]float32, codecLayers*window*qd), make([]float32, codecLayers*window*qd)
	d.scores = make([]float32, window)
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
	maxCol, maxY := 0, 0
	for i := range c.blocks {
		b := &c.blocks[i]
		d.snake[i] = make([]float32, (1+rows)*b.in)
		rows *= b.rate
		d.bOut[i] = make([]float32, rows*b.out)
		for j := range b.units {
			d.unit[i][j] = make([]float32, (b.units[j].conv1.history()+rows)*b.out)
			maxCol = max(maxCol, rows*7*b.out)
		}
		maxY = max(maxY, rows*b.out)
	}
	d.col = make([]float32, maxCol)
	d.y = make([]float32, maxY)
	d.outIn = make([]float32, (c.outConv.history()+rows)*c.outConv.in)
	return d
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
		for j := range d.unit[i] {
			clear(d.unit[i][j])
		}
	}
	clear(d.outIn)
}

// decode turns one frame's codes into FrameSamples samples of pcm.
func (d *decoder) decode(frame *[groups]int, pcm []float32) {
	c := d.c
	// Codebooks: the first and the rest have separate projections.
	copy(d.sum, c.firstTable[frame[0]*codebookDim:(frame[0]+1)*codebookDim])
	c.firstProj.apply(d.exec, d.proj, d.sum, 1)
	pre := d.pre[c.preConv.history()*codecHidden:]
	copy(pre, d.proj)
	clear(d.sum)
	for g := 1; g < groups; g++ {
		addTo(d.sum, c.restTables[((g-1)*codes+frame[g])*codebookDim:][:codebookDim])
	}
	c.restProj.apply(d.exec, d.proj, d.sum, 1)
	addTo(pre[:codecHidden], d.proj)
	d.conv(&c.preConv, d.pre, 1, d.x)
	d.transformer()

	// Two upsamplers: a stride-2 transposed convolution, then ConvNeXt.
	copy(d.upIn[0], d.lat)
	rows := 1
	for i := range c.up {
		u := &c.up[i]
		hist := d.dw[i]
		out := hist[6*latent:]
		u.trans.apply(d.exec, out, d.upIn[i], rows)
		rows *= 2
		d.convNeXt(u, hist, rows, d.upX[i])
		if i+1 < len(c.up) {
			copy(d.upIn[i+1], d.upX[i][:rows*latent])
		}
	}
	copy(d.in0[c.inConv.history()*latent:], d.upX[1][:rows*latent])
	d.conv(&c.inConv, d.in0, rows, d.b0)
	x := d.b0
	for i := range c.blocks {
		b := &c.blocks[i]
		hist := d.snake[i]
		b.snake.apply(hist[b.in:], x[:rows*b.in], b.in)
		out := d.bOut[i]
		// Row t reads [x(t-1), x(t)]: consecutive rows overlap by one input.
		d.exec.Mul(b.trans.w, out, b.trans.out, hist, b.in, rows)
		for r := range rows {
			addTo(out[r*b.trans.out:(r+1)*b.trans.out], b.trans.b)
		}
		// The transposed convolution reads rows t-1 and t: keep the last.
		copy(hist[:b.in], hist[rows*b.in:(rows+1)*b.in])
		rows *= b.rate
		for j := range b.units {
			u := &b.units[j]
			uh := d.unit[i][j]
			u.act1.apply(uh[u.conv1.history()*b.out:], out[:rows*b.out], b.out)
			y := d.y[:rows*b.out]
			d.conv(&u.conv1, uh, rows, y)
			u.act2.apply(y, y, b.out)
			z := d.col[:rows*b.out]
			u.conv2.apply(d.exec, z, y, rows)
			addTo(out[:rows*b.out], z)
		}
		x = out
	}
	oc := &c.outConv
	c.outSnake.apply(d.outIn[oc.history()*oc.in:], x[:rows*oc.in], oc.in)
	d.conv(oc, d.outIn, rows, pcm[:rows])
	for i, v := range pcm[:rows] {
		pcm[i] = min(max(v, -1), 1)
	}
	d.pos++
}

// conv runs a causal convolution over rows new inputs, which follow its
// history in hist, and then keeps the latest history.
func (d *decoder) conv(c *conv, hist []float32, rows int, dst []float32) {
	h := c.history()
	if c.dilation == 1 {
		// Consecutive rows overlap: row t reads hist[t·in : (t+k)·in].
		d.exec.Mul(c.w.w, dst, c.out, hist, c.in, rows)
		for r := range rows {
			addTo(dst[r*c.out:(r+1)*c.out], c.w.b)
		}
	} else {
		col := d.col[:rows*c.k*c.in]
		for t := range rows {
			for j := range c.k {
				copy(col[(t*c.k+j)*c.in:(t*c.k+j+1)*c.in], hist[(t+j*c.dilation)*c.in:(t+j*c.dilation+1)*c.in])
			}
		}
		c.w.apply(d.exec, dst, col, rows)
	}
	copy(hist[:h*c.in], hist[rows*c.in:(rows+h)*c.in])
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
	u.pw1.apply(d.exec, d.pw, d.ln, rows)
	for i, v := range d.pw[:rows*4*latent] {
		d.pw[i] = gelu(v)
	}
	u.pw2.apply(d.exec, d.ln, d.pw, rows)
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
	c := d.c
	c.inProj.apply(d.exec, d.h, d.x, 1)
	qd, half := codecHeads*codecHeadD, codecHeadD/2
	slot := d.pos % window
	span := min(d.pos+1, window)
	cos, sin := c.cos[d.pos*half:(d.pos+1)*half], c.sin[d.pos*half:(d.pos+1)*half]
	scale := float32(1 / math.Sqrt(codecHeadD))
	for i := range c.layers {
		l := &c.layers[i]
		rmsNorm(d.n, d.h, l.inNorm)
		l.q.Mul(d.q, d.n)
		l.k.Mul(d.k, d.n)
		l.v.Mul(d.v, d.n)
		for hd := range codecHeads {
			rope(d.q[hd*codecHeadD:(hd+1)*codecHeadD], cos, sin)
			rope(d.k[hd*codecHeadD:(hd+1)*codecHeadD], cos, sin)
		}
		copy(d.layerK(i)[slot*qd:(slot+1)*qd], d.k)
		copy(d.layerV(i)[slot*qd:(slot+1)*qd], d.v)
		keys, values := d.layerK(i), d.layerV(i)
		for hd := range codecHeads {
			q := d.q[hd*codecHeadD : (hd+1)*codecHeadD]
			top := float32(math.Inf(-1))
			for j := range span {
				p := d.pos - j
				s := p % window
				k := keys[s*qd+hd*codecHeadD : s*qd+(hd+1)*codecHeadD]
				var dot float32
				for e := range codecHeadD {
					dot += q[e] * k[e]
				}
				d.scores[j] = dot * scale
				top = max(top, d.scores[j])
			}
			var sum float32
			for j := range span {
				d.scores[j] = float32(math.Exp(float64(d.scores[j] - top)))
				sum += d.scores[j]
			}
			o := d.att[hd*codecHeadD : (hd+1)*codecHeadD]
			clear(o)
			for j := range span {
				p := d.pos - j
				s := p % window
				w := d.scores[j] / sum
				v := values[s*qd+hd*codecHeadD : s*qd+(hd+1)*codecHeadD]
				for e := range codecHeadD {
					o[e] += w * v[e]
				}
			}
		}
		l.o.Mul(d.o, d.att)
		for e := range codecHidden {
			d.h[e] += l.attnScale[e] * d.o[e]
		}
		rmsNorm(d.n, d.h, l.postNorm)
		l.gate.Mul(d.gate, d.n)
		l.up.Mul(d.up, d.n)
		for e := range codecInter {
			d.gate[e] = silu(d.gate[e]) * d.up[e]
		}
		l.down.Mul(d.o, d.gate)
		for e := range codecHidden {
			d.h[e] += l.mlpScale[e] * d.o[e]
		}
	}
	rmsNorm(d.n, d.h, c.norm)
	c.outProj.apply(d.exec, d.lat, d.n, 1)
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
