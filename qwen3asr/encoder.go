// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/GetStream/gophonic/internal/nn"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

// The audio encoder (AuT) splits the log-mel features into chunks of
// 2·n_window frames, pads every chunk to the longest one with zero
// features, and downsamples each by eight in time and frequency with three
// stride-two 3×3 convolutions and exact GELUs. A linear layer maps each
// output frame's channels×frequencies to the model width, and sinusoidal
// positions restart in every chunk. Only frames that come from real features
// continue, into pre-LayerNorm transformer layers that attend bidirectionally
// within windows of n_window_infer/(2·n_window) chunks. A final LayerNorm and
// a two-layer GELU projection produce one decoder embedding per frame.
//
// Activations are time-major with channels innermost, [time][frequency]
// [channel], so each 3×3 convolution is one matrix product over im2col rows
// whose three frequency taps are contiguous, and the last convolution's
// output rows are the conv_out inputs as they are.
type encoder struct {
	d, heads, headDim, ffn, out int
	layerCount                  int
	ch                          int    // convolution channels
	freq                        [4]int // mel bins, then after each convolution
	chunkFrames                 int    // 2·n_window
	windowChunks                int    // n_window_infer / (2·n_window)
	conv                        [3]*whispergemm.PackedB
	convBias                    [3][]float32
	convOut                     *whispergemm.PackedB
	positions                   []float32 // [tokens per chunk][d]
	layers                      []encoderLayer
	postW, postB                []float32
	proj1, proj2                *whispergemm.PackedB
	proj1B, proj2B              []float32
}

type encoderLayer struct {
	attnW, attnB, ffnW, ffnB []float32 // LayerNorms
	qkv, out, fc1, fc2       *whispergemm.PackedB
	qkvB, outB, fc1B, fc2B   []float32
}

// convLen is the output length of a stride-two, padding-one, size-three
// convolution over n positions.
func convLen(n int) int { return (n + 1) / 2 }

// frameTokens is the number of encoder outputs of a chunk of n frames.
func frameTokens(n int) int { return convLen(convLen(convLen(n))) }

// Tokens returns the number of audio embeddings, and so of <|audio_pad|>
// tokens, for frames log-mel frames.
func (e *encoder) tokens(frames int) int {
	return frames/e.chunkFrames*frameTokens(e.chunkFrames) + frameTokens(frames%e.chunkFrames)
}

// newEncoder validates the configuration and returns the encoder's
// geometry, without weights.
func newEncoder(c audioConfig) (*encoder, error) {
	if c.Activation != "gelu" || c.ScaleEmbed || c.DModel <= 0 || c.Heads <= 0 || c.DModel%c.Heads != 0 ||
		c.Layers <= 0 || c.FFN <= 0 || c.Downsample <= 0 || c.MelBins <= 0 || c.Window <= 0 ||
		c.WindowInfer <= 0 || c.WindowInfer%(2*c.Window) != 0 || c.OutputDim <= 0 {
		return nil, errors.New("qwen3asr: unsupported audio encoder configuration")
	}
	e := &encoder{d: c.DModel, heads: c.Heads, headDim: c.DModel / c.Heads, ffn: c.FFN, out: c.OutputDim,
		ch: c.Downsample, chunkFrames: 2 * c.Window, windowChunks: c.WindowInfer / (2 * c.Window),
		layerCount: c.Layers}
	e.freq[0] = c.MelBins
	for i := 1; i < 4; i++ {
		e.freq[i] = convLen(e.freq[i-1])
	}
	if frameTokens(e.chunkFrames) > c.MaxPositions {
		return nil, errors.New("qwen3asr: audio chunks exceed the sinusoidal positions")
	}
	e.positions = sinusoids(frameTokens(e.chunkFrames), e.d)
	return e, nil
}

// loadEncoder reads the encoder's weights for the CPU.
func loadEncoder(st *safetensors.Checkpoint, c audioConfig, prefix string) (*encoder, error) {
	e, err := newEncoder(c)
	if err != nil {
		return nil, err
	}
	// Load and pack tensors in parallel; each job packs one matrix.
	var (
		mu    sync.Mutex
		first error
		wg    sync.WaitGroup
		jobs  = make(chan func() error)
	)
	for range min(runtime.GOMAXPROCS(0), 8) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				if err := job(); err != nil {
					mu.Lock()
					first = cmp(first, err)
					mu.Unlock()
				}
			}
		}()
	}
	vector := func(name string, n int, dst *[]float32) func() error {
		return func() (err error) {
			*dst, err = st.Float32(prefix+name, n)
			return err
		}
	}
	// linear packs PyTorch [n][k] weights for x·Wᵀ.
	linear := func(names []string, n, k int, dst **whispergemm.PackedB) func() error {
		return func() error {
			rows := make([]float32, 0, len(names)*n*k)
			for _, name := range names {
				w, err := st.Float32(prefix+name, n, k)
				if err != nil {
					return err
				}
				rows = append(rows, w...)
			}
			b, err := whispergemm.NewPackedB(k, len(names)*n)
			if err != nil {
				return err
			}
			*dst = b
			return b.Pack(rows, k, true)
		}
	}
	// conv packs a PyTorch [out][in][3 freq][3 time] kernel with K ordered
	// (time tap, frequency tap, input channel), matching the im2col rows.
	conv := func(i, in int) func() error {
		return func() error {
			name := fmt.Sprintf("%sconv2d%d.", prefix, i+1)
			w, err := st.Float32(name+"weight", e.ch, in, 3, 3)
			if err != nil {
				return err
			}
			if e.convBias[i], err = st.Float32(name+"bias", e.ch); err != nil {
				return err
			}
			k := 9 * in
			rows := make([]float32, e.ch*k)
			for o := range e.ch {
				for c := range in {
					for fi := range 3 {
						for tj := range 3 {
							rows[o*k+(tj*3+fi)*in+c] = w[((o*in+c)*3+fi)*3+tj]
						}
					}
				}
			}
			b, err := whispergemm.NewPackedB(k, e.ch)
			if err != nil {
				return err
			}
			e.conv[i] = b
			return b.Pack(rows, k, true)
		}
	}
	var sends []func() error
	sends = append(sends, conv(0, 1), conv(1, e.ch), conv(2, e.ch))
	// conv_out reads channel-major c·F+f; the activations are f·C+c.
	sends = append(sends, func() error {
		f := e.freq[3]
		k := e.ch * f
		w, err := st.Float32(prefix+"conv_out.weight", e.d, k)
		if err != nil {
			return err
		}
		rows := make([]float32, len(w))
		for o := range e.d {
			for c := range e.ch {
				for fi := range f {
					rows[o*k+fi*e.ch+c] = w[o*k+c*f+fi]
				}
			}
		}
		b, err := whispergemm.NewPackedB(k, e.d)
		if err != nil {
			return err
		}
		e.convOut = b
		return b.Pack(rows, k, true)
	})
	e.layers = make([]encoderLayer, e.layerCount)
	for i := range e.layers {
		l, p := &e.layers[i], fmt.Sprintf("layers.%d.", i)
		sends = append(sends,
			vector(p+"self_attn_layer_norm.weight", e.d, &l.attnW), vector(p+"self_attn_layer_norm.bias", e.d, &l.attnB),
			vector(p+"final_layer_norm.weight", e.d, &l.ffnW), vector(p+"final_layer_norm.bias", e.d, &l.ffnB),
			linear([]string{p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"}, e.d, e.d, &l.qkv),
			linear([]string{p + "self_attn.out_proj.weight"}, e.d, e.d, &l.out),
			linear([]string{p + "fc1.weight"}, e.ffn, e.d, &l.fc1),
			linear([]string{p + "fc2.weight"}, e.d, e.ffn, &l.fc2),
			func() error {
				l.qkvB = make([]float32, 0, 3*e.d)
				for _, name := range []string{"q_proj", "k_proj", "v_proj"} {
					b, err := st.Float32(prefix+p+"self_attn."+name+".bias", e.d)
					if err != nil {
						return err
					}
					l.qkvB = append(l.qkvB, b...)
				}
				return nil
			},
			vector(p+"self_attn.out_proj.bias", e.d, &l.outB),
			vector(p+"fc1.bias", e.ffn, &l.fc1B), vector(p+"fc2.bias", e.d, &l.fc2B))
	}
	sends = append(sends,
		vector("ln_post.weight", e.d, &e.postW), vector("ln_post.bias", e.d, &e.postB),
		linear([]string{"proj1.weight"}, e.d, e.d, &e.proj1), vector("proj1.bias", e.d, &e.proj1B),
		linear([]string{"proj2.weight"}, e.out, e.d, &e.proj2), vector("proj2.bias", e.out, &e.proj2B))
	for _, job := range sends {
		jobs <- job
	}
	close(jobs)
	wg.Wait()
	if first != nil {
		return nil, fmt.Errorf("qwen3asr: audio encoder: %w", first)
	}
	return e, nil
}

func cmp(first, err error) error {
	if first != nil {
		return first
	}
	return err
}

// sinusoids returns rows [0, length) of the encoder's position table in the
// FP32 arithmetic of the reference: sines of position·exp(-i·ln(10⁴)/(C/2-1))
// in the first half of each row, cosines in the second.
func sinusoids(length, channels int) []float32 {
	half := channels / 2
	increment := float32(math.Log(10000) / float64(half-1))
	table := make([]float32, length*channels)
	for i := range half {
		inv := float32(math.Exp(float64(-increment * float32(i))))
		for t := range length {
			angle := float64(float32(t) * inv)
			table[t*channels+i] = float32(math.Sin(angle))
			table[t*channels+half+i] = float32(math.Cos(angle))
		}
	}
	return table
}

// encoderWorkspace owns one lane's encoder activations and workers. Buffers
// grow to the longest audio seen and keep that capacity.
type encoderWorkspace struct {
	exec    *whispergemm.Executor
	workers int
	cols    []float32 // im2col rows of one chunk
	conv    [2][]float32
	stacked []float32 // last convolution output of every chunk, [chunk·time][freq·channel]
	x, norm []float32 // [tokens][d]
	qkv     []float32 // [tokens][3d]
	ctx     []float32 // [tokens][d]
	ffn     []float32 // [tokens][ffn]
	attn    []attentionScratch
	op      encoderRows
}

type attentionScratch struct {
	keysT, values *whispergemm.PackedB
	scores, gemm  []float32
	inverse       []float32
}

func newEncoderWorkspace(e *encoder, workers int) (*encoderWorkspace, error) {
	exec, err := whispergemm.NewExecutor(workers)
	if err != nil {
		return nil, err
	}
	w := &encoderWorkspace{exec: exec, workers: workers, attn: make([]attentionScratch, workers)}
	w.op.w, w.op.e = w, e
	window := frameTokens(e.chunkFrames) * e.windowChunks
	for i := range w.attn {
		a := &w.attn[i]
		if a.keysT, err = whispergemm.NewPackedB(e.headDim, window); err != nil {
			return nil, err
		}
		if a.values, err = whispergemm.NewPackedB(window, e.headDim); err != nil {
			return nil, err
		}
		a.scores = make([]float32, window*window)
		a.inverse = make([]float32, window)
		a.gemm = make([]float32, whispergemm.ScratchLen(max(window, e.headDim)))
	}
	return w, nil
}

func (w *encoderWorkspace) close() {
	if w.exec != nil {
		_ = w.exec.Close()
		w.exec = nil
	}
}

// ensure returns s with length n, reallocating only when it lacks capacity.
func ensure(s []float32, n int) []float32 {
	if cap(s) < n {
		return make([]float32, n)
	}
	return s[:n]
}

// encode runs the encoder on channel-major [bins][frames] features and
// writes [tokens][out] embeddings to dst, returning the token count.
func (e *encoder) encode(mel []float32, frames int, dst []float32, w *encoderWorkspace) (int, error) {
	if frames <= 0 || len(mel) != e.freq[0]*frames {
		return 0, errors.New("qwen3asr: feature shape does not match the encoder")
	}
	chunks := (frames + e.chunkFrames - 1) / e.chunkFrames
	span := e.chunkFrames // every chunk is padded to the longest
	if chunks == 1 {
		span = frames
	}
	t1, t2, t3 := convLen(span), convLen(convLen(span)), frameTokens(span)
	f1, f2, f3 := e.freq[1], e.freq[2], e.freq[3]
	n := e.tokens(frames)
	if len(dst) < n*e.out {
		return 0, errors.New("qwen3asr: encoder output buffer too short")
	}
	w.cols = ensure(w.cols, max(t1*f1*9, t2*f2*9*e.ch))
	w.conv[0] = ensure(w.conv[0], t1*f1*e.ch)
	w.conv[1] = ensure(w.conv[1], t2*f2*e.ch)
	w.stacked = ensure(w.stacked, chunks*t3*f3*e.ch)
	op := &w.op
	for c := range chunks {
		start := c * e.chunkFrames
		real := min(e.chunkFrames, frames-start)
		*op = encoderRows{w: w, e: e, kind: rowsMelColumns, src: mel, frames: frames, start: start, real: real, width: f1, inTime: span}
		if err := w.rows(t1 * f1); err != nil {
			return 0, err
		}
		if err := w.exec.Mul(e.conv[0], w.conv[0], e.ch, w.cols, 9, t1*f1); err != nil {
			return 0, err
		}
		if err := w.gelu(w.conv[0], e.convBias[0], t1*f1, e.ch); err != nil {
			return 0, err
		}
		layers := [2]struct {
			src, dst          []float32
			inTime, inF, outF int
			outTime           int
		}{{w.conv[0], w.conv[1], t1, f1, f2, t2}, {w.conv[1], w.stacked[c*t3*f3*e.ch:], t2, f2, f3, t3}}
		for i, l := range layers {
			*op = encoderRows{w: w, e: e, kind: rowsConvColumns, src: l.src, inTime: l.inTime, inFreq: l.inF, width: l.outF}
			if err := w.rows(l.outTime * l.outF); err != nil {
				return 0, err
			}
			if err := w.exec.Mul(e.conv[i+1], l.dst, e.ch, w.cols, 9*e.ch, l.outTime*l.outF); err != nil {
				return 0, err
			}
			if err := w.gelu(l.dst[:l.outTime*l.outF*e.ch], e.convBias[i+1], l.outTime*l.outF, e.ch); err != nil {
				return 0, err
			}
		}
	}
	d := e.d
	w.x = ensure(w.x, max(chunks*t3, n)*d)
	w.norm = ensure(w.norm, n*d)
	w.qkv = ensure(w.qkv, n*3*d)
	w.ctx = ensure(w.ctx, n*d)
	w.ffn = ensure(w.ffn, n*max(e.ffn, d))
	if err := w.exec.Mul(e.convOut, w.x, d, w.stacked, f3*e.ch, chunks*t3); err != nil {
		return 0, err
	}
	// Keep each chunk's real frames, in place, and add their positions.
	row := 0
	for c := range chunks {
		keep := frameTokens(min(e.chunkFrames, frames-c*e.chunkFrames))
		for t := range keep {
			dstRow, srcRow := w.x[row*d:(row+1)*d], w.x[(c*t3+t)*d:(c*t3+t+1)*d]
			pos := e.positions[t*d : (t+1)*d]
			for i := range dstRow {
				dstRow[i] = srcRow[i] + pos[i]
			}
			row++
		}
	}
	x := w.x[:n*d]
	*op = encoderRows{w: w, e: e, kind: rowsNorm, dst: x, out: w.norm, normW: e.layers[0].attnW, normB: e.layers[0].attnB, width: d}
	if err := w.rows(n); err != nil {
		return 0, err
	}
	window := frameTokens(min(span, e.chunkFrames)) * e.windowChunks
	if chunks > 1 {
		window = frameTokens(e.chunkFrames) * e.windowChunks
	}
	for i := range e.layers {
		l := &e.layers[i]
		nextW, nextB := e.postW, e.postB
		if i+1 < len(e.layers) {
			nextW, nextB = e.layers[i+1].attnW, e.layers[i+1].attnB
		}
		if err := w.exec.Mul(l.qkv, w.qkv, 3*d, w.norm, d, n); err != nil {
			return 0, err
		}
		*op = encoderRows{w: w, e: e, kind: rowsQKV, dst: w.qkv, bias: l.qkvB, width: 3 * d}
		if err := w.rows(n); err != nil {
			return 0, err
		}
		if err := w.attention(n, window); err != nil {
			return 0, err
		}
		if err := w.exec.Mul(l.out, w.norm, d, w.ctx, d, n); err != nil {
			return 0, err
		}
		*op = encoderRows{w: w, e: e, kind: rowsResidualNorm, dst: x, out: w.norm, bias: l.outB, normW: l.ffnW, normB: l.ffnB, width: d}
		if err := w.rows(n); err != nil {
			return 0, err
		}
		if err := w.exec.Mul(l.fc1, w.ffn, e.ffn, w.norm, d, n); err != nil {
			return 0, err
		}
		if err := w.gelu(w.ffn[:n*e.ffn], l.fc1B, n, e.ffn); err != nil {
			return 0, err
		}
		if err := w.exec.Mul(l.fc2, w.norm, d, w.ffn, e.ffn, n); err != nil {
			return 0, err
		}
		*op = encoderRows{w: w, e: e, kind: rowsResidualNorm, dst: x, out: w.norm, bias: l.fc2B, normW: nextW, normB: nextB, width: d}
		if err := w.rows(n); err != nil {
			return 0, err
		}
	}
	// w.norm holds ln_post(x).
	if err := w.exec.Mul(e.proj1, w.ffn, d, w.norm, d, n); err != nil {
		return 0, err
	}
	if err := w.gelu(w.ffn[:n*d], e.proj1B, n, d); err != nil {
		return 0, err
	}
	if err := w.exec.Mul(e.proj2, dst, e.out, w.ffn, d, n); err != nil {
		return 0, err
	}
	*op = encoderRows{w: w, e: e, kind: rowsBias, dst: dst, bias: e.proj2B, width: e.out}
	return n, w.rows(n)
}

func (w *encoderWorkspace) rows(n int) error {
	err := w.exec.Rows(&w.op, n, 8)
	w.op.src, w.op.dst, w.op.out, w.op.bias = nil, nil, nil, nil
	return err
}

func (w *encoderWorkspace) gelu(values, bias []float32, rows, width int) error {
	w.op = encoderRows{w: w, e: w.op.e, kind: rowsGELU, dst: values, bias: bias, width: width}
	return w.rows(rows)
}

// attention runs windowed self-attention over the n rows of w.qkv, which
// hold scaled queries, keys, and values, and writes the context to w.ctx.
// Workers claim (window, head) items.
func (w *encoderWorkspace) attention(n, window int) error {
	e := w.op.e
	windows := (n + window - 1) / window
	w.op = encoderRows{w: w, e: e, kind: rowsAttention, rows: n, window: window, items: windows * e.heads}
	w.op.next.Store(0)
	return w.exec.Rows(&w.op, w.workers, 1)
}

type encoderRowKind uint8

const (
	rowsMelColumns encoderRowKind = iota
	rowsConvColumns
	rowsGELU
	rowsNorm
	rowsQKV
	rowsResidualNorm
	rowsBias
	rowsAttention
)

// encoderRows is the encoder's one reusable parallel row operation; it is
// stored in the workspace so dispatch does not allocate.
type encoderRows struct {
	w                  *encoderWorkspace
	e                  *encoder
	kind               encoderRowKind
	src, dst, out      []float32
	bias, normW, normB []float32
	width              int
	// Im2col geometry: input time and frequency extents; for mel columns,
	// the feature frame count, the chunk's first frame, and its real frames.
	inTime, inFreq, frames, start, real int
	// Attention: rows, window length, (window, head) items, and the next
	// unclaimed item.
	rows, window, items int
	next                atomic.Int64
}

func (op *encoderRows) ApplyRows(start, end int) {
	e, w := op.e, op.width
	switch op.kind {
	case rowsMelColumns:
		// Row (t, f) of the first convolution: the nine features around
		// (2t, 2f) in (time tap, frequency tap) order. Frames past the
		// chunk's real ones are the zero padding of the reference.
		bins := e.freq[0]
		cols := op.w.cols
		for r := start; r < end; r++ {
			t, f := r/w, r%w
			dst := cols[r*9 : r*9+9]
			for tj := range 3 {
				ti := 2*t - 1 + tj
				for fi := range 3 {
					fb := 2*f - 1 + fi
					v := float32(0)
					if ti >= 0 && ti < op.real && fb >= 0 && fb < bins {
						v = op.src[fb*op.frames+op.start+ti]
					}
					dst[tj*3+fi] = v
				}
			}
		}
	case rowsConvColumns:
		// Row (t, f): for each time tap, the three frequency taps of every
		// channel, contiguous in the [time][freq][channel] input.
		ch := e.ch
		k := 9 * ch
		for r := start; r < end; r++ {
			t, f := r/w, r%w
			dst := op.w.cols[r*k : (r+1)*k]
			for tj := range 3 {
				seg := dst[tj*3*ch : (tj+1)*3*ch]
				ti := 2*t - 1 + tj
				if ti < 0 || ti >= op.inTime {
					clear(seg)
					continue
				}
				row := op.src[ti*op.inFreq*ch:]
				f0 := 2*f - 1
				if f0 < 0 {
					clear(seg[:ch])
					copy(seg[ch:], row[:2*ch])
				} else if f0+3 > op.inFreq {
					copy(seg, row[f0*ch:op.inFreq*ch])
					clear(seg[(op.inFreq-f0)*ch:])
				} else {
					copy(seg, row[f0*ch:(f0+3)*ch])
				}
			}
		}
	case rowsGELU:
		nn.BiasGELU(op.dst[start*w:end*w], op.bias, end-start, w)
	case rowsNorm:
		for r := start; r < end; r++ {
			nn.LayerNorm(op.dst[r*w:(r+1)*w], op.out[r*w:(r+1)*w], op.normW, op.normB)
		}
	case rowsQKV:
		// Add the biases and fold the 1/sqrt(head) score scale into the
		// queries; the scale is a power of two for every Qwen3-ASR size, so
		// this equals scaling the scores.
		d := e.d
		scale := float32(1 / math.Sqrt(float64(e.headDim)))
		for r := start; r < end; r++ {
			row := op.dst[r*w : (r+1)*w]
			for i, b := range op.bias[:d] {
				row[i] = (row[i] + b) * scale
			}
			nn.AddRowBias(row[d:], op.bias[d:], 1, 2*d)
		}
	case rowsResidualNorm:
		for r := start; r < end; r++ {
			row := op.out[r*w : (r+1)*w]
			nn.ResidualNorm(op.dst[r*w:(r+1)*w], row, op.normW, op.normB, row, op.bias)
		}
	case rowsBias:
		nn.AddRowBias(op.dst[start*w:end*w], op.bias, end-start, w)
	case rowsAttention:
		for worker := start; worker < end; worker++ {
			for {
				item := int(op.next.Add(1) - 1)
				if item >= op.items {
					break
				}
				op.attend(&op.w.attn[worker], item/e.heads, item%e.heads)
			}
		}
	}
}

// attend computes one head of one window: softmax(Q·Kᵀ)·V with the queries
// already scaled, normalizing each row after the value product.
func (op *encoderRows) attend(a *attentionScratch, window, head int) {
	e, w := op.e, op.w
	d, hd := e.d, e.headDim
	r0 := window * op.window
	n := min(op.window, op.rows-r0)
	qkv := w.qkv[r0*3*d:]
	must(a.keysT.Reshape(hd, n))
	must(a.keysT.Pack(qkv[d+head*hd:], 3*d, true))
	scores := a.scores[:n*n]
	must(a.keysT.MulScratch(scores, n, qkv[head*hd:], 3*d, n, a.gemm))
	for r := range n {
		a.inverse[r] = nn.SoftmaxExp(scores[r*n : (r+1)*n])
	}
	must(a.values.Reshape(n, hd))
	must(a.values.Pack(qkv[2*d+head*hd:], 3*d, false))
	ctx := w.ctx[r0*d+head*hd:]
	must(a.values.MulScratch(ctx, d, scores, n, n, a.gemm))
	for r := range n {
		row := ctx[r*d : r*d+hd]
		inv := a.inverse[r]
		for i := range row {
			row[i] *= inv
		}
	}
}

// must panics on a shape error, which validated geometry rules out.
func must(err error) {
	if err != nil {
		panic("qwen3asr: " + err.Error())
	}
}
