// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"fmt"
	"math"

	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

// The codec decoder of Qwen3-TTS-Tokenizer-12Hz turns each frame's sixteen
// codes into 1920 samples. Every layer is causal: convolutions look back
// only, transposed convolutions overlap forward by one input, and the
// transformer attends to the last 72 frames. So a decoder that keeps each
// layer's left context decodes frame by frame and produces exactly the
// samples the reference decodes from the whole sequence at once.
//
// Activations are time-major, one row of channels per time step, so a
// convolution over a history of rows is one matrix product: with dilation
// one its input rows overlap in memory, and otherwise they are gathered.
const (
	codebookDim = 256
	latent      = 1024
	codecHidden = 512
	codecHeads  = 16
	codecHeadD  = 64
	codecInter  = 1024
	codecLayers = 8
	window      = 72 // the transformer's attention span, in frames
	codecEps    = 1e-5
)

var upsampleRates = [4]int{8, 5, 4, 3}

// codec holds the decoder's weights, shared by every lane.
type codec struct {
	firstTable, restTables []float32 // [codes][codebookDim] per codebook, normalized
	firstProj, restProj    dense     // codebookDim → 512
	preConv                conv      // 512 → 1024, kernel 3
	inProj, outProj        dense
	layers                 [codecLayers]codecLayer
	norm                   []float32
	up                     [2]upsampler
	inConv                 conv // 1024 → 1536, kernel 7
	blocks                 [4]block
	outSnake               snake
	outConv                conv // 96 → 1, kernel 7
	threads                int
	cos, sin               []float32 // RoPE, [position][codecHeadD/2]
}

type codecLayer struct {
	inNorm, postNorm []float32
	q, k, v, o       *whispergemm.PackedVector
	gate, up, down   *whispergemm.PackedVector
	attnScale        []float32
	mlpScale         []float32
}

type upsampler struct {
	trans dense // 1024 → 2·1024: both output steps of one input row
	dw    []float32
	dwB   []float32 // depthwise causal kernel 7, [channel][7]
	normW []float32
	normB []float32
	pw1   dense // 1024 → 4096
	pw2   dense // 4096 → 1024
	gamma []float32
}

type block struct {
	snake snake
	trans dense // [x(t-1), x(t)] → r output rows
	rate  int
	in    int
	out   int
	units [3]resUnit
}

type resUnit struct {
	act1, act2 snake
	conv1      conv // kernel 7, dilated
	conv2      dense
}

// snake is SnakeBeta: x + sin²(a·x)/b, with a = exp(α) and 1/b =
// 1/(exp(β)+1e-9) precomputed.
type snake struct{ a, invB []float32 }

// conv is a causal convolution: out[t] = W·[x(t-(k-1)d), …, x(t)] + b.
type conv struct {
	w                    dense // k·in → out
	k, dilation, in, out int
}

func (c conv) history() int { return (c.k - 1) * c.dilation }

func loadCodec(dir string, threads int) (*codec, error) {
	st, err := safetensors.Open(dir)
	if err != nil {
		return nil, err
	}
	defer st.Close()
	d := &codec{threads: threads}
	p := "decoder."
	// Codebooks: embedding = sum / max(usage, eps).
	table := func(name string) ([]float32, error) {
		sum, err := st.Float32(name+"._codebook.embedding_sum", codes, codebookDim)
		if err != nil {
			return nil, err
		}
		usage, err := st.Float32(name+"._codebook.cluster_usage", codes)
		if err != nil {
			return nil, err
		}
		for i := range codes {
			inv := 1 / max(usage[i], 1e-5)
			for j := range codebookDim {
				sum[i*codebookDim+j] *= inv
			}
		}
		return sum, nil
	}
	if d.firstTable, err = table(p + "quantizer.rvq_first.vq.layers.0"); err != nil {
		return nil, err
	}
	for g := range groups - 1 {
		t, err := table(fmt.Sprintf("%squantizer.rvq_rest.vq.layers.%d", p, g))
		if err != nil {
			return nil, err
		}
		d.restTables = append(d.restTables, t...)
	}
	if d.firstProj, err = loadDense(st, p+"quantizer.rvq_first.output_proj", codecHidden, codebookDim, codecHidden, codebookDim, 1); err != nil {
		return nil, err
	}
	if d.restProj, err = loadDense(st, p+"quantizer.rvq_rest.output_proj", codecHidden, codebookDim, codecHidden, codebookDim, 1); err != nil {
		return nil, err
	}
	if d.preConv, err = loadConv(st, p+"pre_conv.conv", codecHidden, latent, 3, 1); err != nil {
		return nil, err
	}
	if d.inProj, err = loadDense(st, p+"pre_transformer.input_proj", codecHidden, latent); err != nil {
		return nil, err
	}
	if d.outProj, err = loadDense(st, p+"pre_transformer.output_proj", latent, codecHidden); err != nil {
		return nil, err
	}
	if d.norm, err = st.Float32(p+"pre_transformer.norm.weight", codecHidden); err != nil {
		return nil, err
	}
	vec := func(name string, rows, k int) (*whispergemm.PackedVector, error) {
		w, err := st.Float32(name, rows, k)
		if err != nil {
			return nil, err
		}
		return whispergemm.NewPackedVector(w, k, rows, k)
	}
	qd := codecHeads * codecHeadD
	for i := range d.layers {
		l, lp := &d.layers[i], fmt.Sprintf("%spre_transformer.layers.%d.", p, i)
		for _, v := range []struct {
			dst  **whispergemm.PackedVector
			name string
			n, k int
		}{{&l.q, "self_attn.q_proj.weight", qd, codecHidden}, {&l.k, "self_attn.k_proj.weight", qd, codecHidden},
			{&l.v, "self_attn.v_proj.weight", qd, codecHidden}, {&l.o, "self_attn.o_proj.weight", codecHidden, qd},
			{&l.gate, "mlp.gate_proj.weight", codecInter, codecHidden}, {&l.up, "mlp.up_proj.weight", codecInter, codecHidden},
			{&l.down, "mlp.down_proj.weight", codecHidden, codecInter}} {
			if *v.dst, err = vec(lp+v.name, v.n, v.k); err != nil {
				return nil, err
			}
		}
		for _, v := range []struct {
			dst  *[]float32
			name string
		}{{&l.inNorm, "input_layernorm.weight"}, {&l.postNorm, "post_attention_layernorm.weight"},
			{&l.attnScale, "self_attn_layer_scale.scale"}, {&l.mlpScale, "mlp_layer_scale.scale"}} {
			if *v.dst, err = st.Float32(lp+v.name, codecHidden); err != nil {
				return nil, err
			}
		}
	}
	for i := range d.up {
		u, up := &d.up[i], fmt.Sprintf("%supsample.%d.", p, i)
		if u.trans, err = loadTransposed(st, up+"0.conv", latent, latent, 2, 2); err != nil {
			return nil, err
		}
		if u.dw, err = st.Float32(up+"1.dwconv.conv.weight", latent, 1, 7); err != nil {
			return nil, err
		}
		for _, v := range []struct {
			dst  *[]float32
			name string
			n    int
		}{{&u.dwB, "1.dwconv.conv.bias", latent}, {&u.normW, "1.norm.weight", latent}, {&u.normB, "1.norm.bias", latent},
			{&u.gamma, "1.gamma", latent}} {
			if *v.dst, err = st.Float32(up+v.name, v.n); err != nil {
				return nil, err
			}
		}
		if u.pw1, err = loadDense(st, up+"1.pwconv1", 4*latent, latent); err != nil {
			return nil, err
		}
		if u.pw2, err = loadDense(st, up+"1.pwconv2", latent, 4*latent); err != nil {
			return nil, err
		}
	}
	const decoderDim = 1536
	if d.inConv, err = loadConv(st, p+"decoder.0.conv", latent, decoderDim, 7, 1); err != nil {
		return nil, err
	}
	for i := range d.blocks {
		b, bp := &d.blocks[i], fmt.Sprintf("%sdecoder.%d.block.", p, i+1)
		b.in, b.out, b.rate = decoderDim>>i, decoderDim>>(i+1), upsampleRates[i]
		if b.snake, err = loadSnake(st, bp+"0", b.in); err != nil {
			return nil, err
		}
		if b.trans, err = loadTransposed(st, bp+"1.conv", b.in, b.out, 2*b.rate, b.rate); err != nil {
			return nil, err
		}
		for j, dilation := range []int{1, 3, 9} {
			u, up := &b.units[j], fmt.Sprintf("%s%d.", bp, j+2)
			if u.act1, err = loadSnake(st, up+"act1", b.out); err != nil {
				return nil, err
			}
			if u.act2, err = loadSnake(st, up+"act2", b.out); err != nil {
				return nil, err
			}
			if u.conv1, err = loadConv(st, up+"conv1.conv", b.out, b.out, 7, dilation); err != nil {
				return nil, err
			}
			if u.conv2, err = loadDense(st, up+"conv2.conv", b.out, b.out, b.out, b.out, 1); err != nil {
				return nil, err
			}
		}
	}
	out := decoderDim >> len(d.blocks)
	if d.outSnake, err = loadSnake(st, p+"decoder.5", out); err != nil {
		return nil, err
	}
	if d.outConv, err = loadConv(st, p+"decoder.6.conv", out, 1, 7, 1); err != nil {
		return nil, err
	}
	// RoPE for positions up to the codec's limit, one per frame.
	const positions = 8000
	half := codecHeadD / 2
	d.cos, d.sin = make([]float32, positions*half), make([]float32, positions*half)
	for pos := range positions {
		for i := range half {
			theta := float64(pos) / math.Pow(10000, float64(2*i)/codecHeadD)
			d.cos[pos*half+i], d.sin[pos*half+i] = float32(math.Cos(theta)), float32(math.Sin(theta))
		}
	}
	return d, nil
}

// loadConv reads a causal Conv1d weight [out][in][k] as a k·in → out
// dense layer whose input rows are k consecutive (dilated) time steps.
func loadConv(st *safetensors.Checkpoint, name string, in, out, k, dilation int) (conv, error) {
	w, err := st.Float32(name+".weight", out, in, k)
	if err != nil {
		return conv{}, err
	}
	// Row co of the transposed B holds W[co][ci][j] at column j·in+ci.
	t := make([]float32, out*k*in)
	for co := range out {
		for ci := range in {
			for j := range k {
				t[co*k*in+j*in+ci] = w[(co*in+ci)*k+j]
			}
		}
	}
	c := conv{k: k, dilation: dilation, in: in, out: out}
	c.w = dense{in: k * in, out: out}
	if c.w.w, err = whispergemm.NewPackedB(k*in, out); err != nil {
		return conv{}, err
	}
	if err := c.w.w.Pack(t, k*in, true); err != nil {
		return conv{}, err
	}
	c.w.b, err = st.Float32(name+".bias", out)
	return c, err
}

// loadTransposed reads a ConvTranspose1d weight [in][out][2r] with stride r
// (or [in][out][r] with kernel equal to the stride) as a dense layer from
// the rows [x(t-1), x(t)] (or [x(t)]) to r output rows.
func loadTransposed(st *safetensors.Checkpoint, name string, in, out, k, r int) (dense, error) {
	w, err := st.Float32(name+".weight", in, out, k)
	if err != nil {
		return dense{}, err
	}
	taps := k / r // 1 or 2 input rows per output block
	kin := taps * in
	t := make([]float32, r*out*kin) // transposed: [j·out+co][tap·in+ci]
	for ci := range in {
		for co := range out {
			for j := range r {
				// Input t contributes W[:,:,j] to its own block and
				// W[:,:,r+j] to the next one; the older row comes first.
				t[(j*out+co)*kin+(taps-1)*in+ci] = w[(ci*out+co)*k+j]
				if taps == 2 {
					t[(j*out+co)*kin+ci] = w[(ci*out+co)*k+r+j]
				}
			}
		}
	}
	d := dense{in: kin, out: r * out}
	if d.w, err = whispergemm.NewPackedB(kin, r*out); err != nil {
		return dense{}, err
	}
	if err := d.w.Pack(t, kin, true); err != nil {
		return dense{}, err
	}
	bias, err := st.Float32(name+".bias", out)
	if err != nil {
		return dense{}, err
	}
	d.b = make([]float32, r*out)
	for j := range r {
		copy(d.b[j*out:], bias)
	}
	return d, nil
}

func loadSnake(st *safetensors.Checkpoint, name string, n int) (snake, error) {
	alpha, err := st.Float32(name+".alpha", n)
	if err != nil {
		return snake{}, err
	}
	beta, err := st.Float32(name+".beta", n)
	if err != nil {
		return snake{}, err
	}
	s := snake{a: make([]float32, n), invB: make([]float32, n)}
	for i := range n {
		s.a[i] = float32(math.Exp(float64(alpha[i])))
		s.invB[i] = float32(1 / (math.Exp(float64(beta[i])) + 1e-9))
	}
	return s, nil
}
