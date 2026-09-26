// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"bufio"
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

// GPTQ weights. QuantizeGPTQ rounds the GPU formats' projections with GPTQ
// (Frantar et al., 2022): each weight column is rounded in turn and the
// rounding error is spread over the columns not yet rounded, weighted by the
// inverse Hessian of the projection's inputs on calibration text. The
// result is stored next to the checkpoint and picked up by LoadWeights.

//go:embed calibration.txt
var calibrationText string

const (
	gptqMagic        = "GPHQ0001"
	gptqBlock        = 128  // columns per lazy update
	gptqDamp         = 0.01 // Hessian damping, relative to its mean diagonal
	gptqMaxRows      = 4096 // calibration tokens
	gptqMaxSeq       = 256  // tokens per calibration sequence
	q4GroupSize      = 32
	gptqFileTemplate = "gophonic-gptq-%s.bin"
)

// gptqProjections lists a layer's projections in file order.
var gptqProjections = [...]string{
	"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
	"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}

// gptqPath returns the GPTQ file for format in the snapshot directory.
func gptqPath(dir, format string) string {
	return filepath.Join(dir, fmt.Sprintf(gptqFileTemplate, format))
}

// snapshotFingerprint hashes the names, sizes, and modification times of the
// checkpoint files, so a GPTQ file is ignored after the checkpoint changes.
func snapshotFingerprint(dir string) (uint64, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}
	var names []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".safetensors") || e.Name() == "config.json" {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	h := fnv.New64a()
	for _, n := range names {
		info, err := os.Stat(filepath.Join(dir, n))
		if err != nil {
			return 0, err
		}
		fmt.Fprintf(h, "%s %d %d\n", n, info.Size(), info.ModTime().UnixNano())
	}
	return h.Sum64(), nil
}

// projectionShape returns a projection's rows and input width.
func projectionShape(c *modelConfig, p int) (n, k int) {
	qdim := c.heads * c.headDim
	switch p {
	case 0:
		return qdim, c.hidden
	case 1, 2:
		return c.kvDim, c.hidden
	case 3:
		return c.hidden, qdim
	case 4, 5:
		return c.intermediate, c.hidden
	}
	return c.hidden, c.intermediate
}

// gptqRowBytes returns the code and scale bytes of one quantized row.
func gptqRowBytes(bits, k int) (codes, scales int) {
	if bits == 4 {
		return k / 2, 2 * (k / q4GroupSize)
	}
	return k, 4
}

// gptqFile is an opened GPTQ weight file.
type gptqFile struct {
	f       *os.File
	bits    int
	c       *modelConfig
	offsets []int64 // per layer and projection
}

// openGPTQ opens dir's GPTQ file for format, or returns nil if there is none
// or it was made from other checkpoint files.
func openGPTQ(dir, format string, bits int, c *modelConfig) *gptqFile {
	f, err := os.Open(gptqPath(dir, format))
	if err != nil {
		return nil
	}
	var head [24]byte
	fp, err := snapshotFingerprint(dir)
	if _, rerr := io.ReadFull(f, head[:]); rerr != nil || err != nil || string(head[:8]) != gptqMagic ||
		binary.LittleEndian.Uint64(head[8:]) != uint64(bits) || binary.LittleEndian.Uint64(head[16:]) != fp {
		f.Close()
		return nil
	}
	g := &gptqFile{f: f, bits: bits, c: c}
	off := int64(len(head))
	for range c.layers {
		for p := range gptqProjections {
			g.offsets = append(g.offsets, off)
			n, k := projectionShape(c, p)
			cb, sb := gptqRowBytes(bits, k)
			off += int64(n * (cb + sb))
		}
	}
	if info, err := f.Stat(); err != nil || info.Size() != off {
		f.Close()
		return nil
	}
	return g
}

// projection locates a tensor's codes and scales from its name after the
// decoder prefix ("layers.3.mlp.up_proj.weight"): offset of the codes,
// offset of the scales, and whether the file holds it.
func (g *gptqFile) projection(name string) (codes, scales int64, ok bool) {
	var layer int
	var rest string
	if _, err := fmt.Sscanf(name, "layers.%d.%s", &layer, &rest); err != nil || layer < 0 || layer >= g.c.layers {
		return 0, 0, false
	}
	rest = strings.TrimSuffix(rest, ".weight")
	for p, proj := range gptqProjections {
		if proj == rest {
			n, k := projectionShape(g.c, p)
			cb, _ := gptqRowBytes(g.bits, k)
			off := g.offsets[layer*len(gptqProjections)+p]
			return off, off + int64(n*cb), true
		}
	}
	return 0, 0, false
}

func (g *gptqFile) close() { g.f.Close() }

// place copies a projection's codes and scales into a GPU layer buffer:
// source row r lands at row row0+r·step of the codes at base and of the
// scales at sc. It reports false if the file does not hold the projection.
func (g *gptqFile) place(name string, n, k int, buf []byte, base, sc, row0, step int) (bool, error) {
	codesAt, scalesAt, ok := g.projection(name)
	if !ok {
		return false, nil
	}
	cb, sb := gptqRowBytes(g.bits, k)
	codes, scales := make([]byte, n*cb), make([]byte, n*sb)
	if _, err := g.f.ReadAt(codes, codesAt); err != nil {
		return false, err
	}
	if _, err := g.f.ReadAt(scales, scalesAt); err != nil {
		return false, err
	}
	for r := range n {
		dst := row0 + r*step
		copy(buf[base+dst*cb:base+(dst+1)*cb], codes[r*cb:(r+1)*cb])
		copy(buf[sc+dst*sb:sc+(dst+1)*sb], scales[r*sb:(r+1)*sb])
	}
	return true, nil
}

// QuantizeGPTQ writes GPTQ-rounded weights for format (WeightsGPU or
// WeightsGPUQ4) next to the checkpoint in dir. It runs the model layer by
// layer in FP32 on built-in calibration text, so each layer is rounded
// against the inputs produced by the already rounded layers before it.
// progress, if not nil, is called after each layer. LoadWeights uses the
// file automatically while the checkpoint files are unchanged.
func QuantizeGPTQ(dir, format string, progress func(layer, layers int)) error {
	bits := 8
	switch format {
	case WeightsGPU:
	case WeightsGPUQ4:
		bits = 4
	default:
		return fmt.Errorf("qwen3: GPTQ supports %q and %q, not %q", WeightsGPU, WeightsGPUQ4, format)
	}
	cfg, err := readConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		return err
	}
	c := &cfg
	tok, err := LoadTokenizer(dir)
	if err != nil {
		return err
	}
	seqs, err := calibrationSequences(tok)
	if err != nil {
		return err
	}
	st, err := safetensors.Open(dir)
	if err != nil {
		return err
	}
	defer st.Close()
	fp, err := snapshotFingerprint(dir)
	if err != nil {
		return err
	}
	tmp := gptqPath(dir, format) + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer os.Remove(tmp)
	w := bufio.NewWriterSize(out, 1<<22)
	var head [24]byte
	copy(head[:], gptqMagic)
	binary.LittleEndian.PutUint64(head[8:], uint64(bits))
	binary.LittleEndian.PutUint64(head[16:], fp)
	w.Write(head[:])

	cal, err := newCalibration(st, c, seqs)
	if err != nil {
		out.Close()
		return err
	}
	for layer := range c.layers {
		if err := cal.layer(st, layer, bits, w); err != nil {
			out.Close()
			return fmt.Errorf("qwen3: GPTQ layer %d: %w", layer, err)
		}
		if progress != nil {
			progress(layer+1, c.layers)
		}
	}
	if err := w.Flush(); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, gptqPath(dir, format))
}

// calibrationSequences tokenizes the embedded corpus into sequences of up to
// gptqMaxSeq tokens, and adds Question prompts over its short lines, the
// shape of most classification inputs.
func calibrationSequences(tok *Tokenizer) ([][]int, error) {
	var ws TokenizerWorkspace
	var seqs [][]int
	total := 0
	add := func(text string) error {
		ids, err := tok.EncodeInto(text, make([]int, 0, 4*len(text)+8), &ws)
		if err != nil {
			return err
		}
		for len(ids) > 0 && total < gptqMaxRows {
			n := min(len(ids), gptqMaxSeq, gptqMaxRows-total)
			seqs = append(seqs, ids[:n])
			total += n
			ids = ids[n:]
		}
		return nil
	}
	for _, para := range strings.Split(calibrationText, "\n\n") {
		if err := add(para); err != nil {
			return nil, err
		}
	}
	// Calibration prompts take the shape of qwen3.Question prompts, the
	// text the GPU path serves most.
	const (
		calibrationHeader = "<|im_start|>user\n"
		calibrationFooter = "Answer with the letter only.\n\nInput:\n"
		calibrationSuffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
	)
	questions := []struct {
		q    string
		opts []string
	}{
		{"What is the sentiment of this message?", []string{"positive", "negative", "neutral"}},
		{"Which team should handle this message?", []string{"billing", "shipping", "technical support", "account", "sales"}},
		{"A voice assistant hears this live transcript. Has the user finished their turn?", []string{"reply now", "wait"}},
		{"What kind of text is this?", []string{"conversation", "code", "instructions", "story", "data"}},
	}
	lines := strings.Split(calibrationText, "\n")
	for i := 0; total < gptqMaxRows && i < 64*len(questions); i++ {
		line := strings.TrimSpace(lines[(i*7)%len(lines)])
		if line == "" || len(line) > 200 {
			continue
		}
		q := questions[i%len(questions)]
		text := calibrationHeader + q.q + "\n"
		for j, o := range q.opts {
			text += string(rune('A'+j)) + ") " + o + "\n"
		}
		if err := add(text + calibrationFooter + line + calibrationSuffix); err != nil {
			return nil, err
		}
	}
	if total == 0 {
		return nil, errors.New("qwen3: empty calibration set")
	}
	return seqs, nil
}

// calibration runs the model in FP32 in the GPU backend's basis: the
// residual stream is rotated by hidden, RMSNorm weights are folded into the
// next projections, value rows and o inputs carry a per-head rotation, and
// down's inputs are rotated by inter.
type calibration struct {
	c                   *modelConfig
	seqs                [][]int
	rows                int
	h                   []float32 // residual stream, [rows][hidden]
	pos, start          []int     // each row's position and sequence start row
	hidden, inter, head *rotation
	cos, sin            []float32 // RoPE tables, [position][headDim/2]
}

func newCalibration(st *safetensors.Checkpoint, c *modelConfig, seqs [][]int) (*calibration, error) {
	cal := &calibration{c: c, seqs: seqs}
	for _, s := range seqs {
		cal.rows += len(s)
	}
	cal.hidden, cal.inter = newRotation(c.hidden), newRotation(c.intermediate)
	head := newRotation(c.headDim)
	cal.head = &rotation{signs: make([]float32, c.heads*c.headDim), block: c.headDim, scale: head.scale}
	for i := range cal.head.signs {
		cal.head.signs[i] = head.signs[i%c.headDim]
	}
	half := c.headDim / 2
	cal.cos, cal.sin = make([]float32, gptqMaxSeq*half), make([]float32, gptqMaxSeq*half)
	for p := range gptqMaxSeq {
		for d, inv := range c.invFreq {
			cal.cos[p*half+d] = float32(math.Cos(float64(p) * inv))
			cal.sin[p*half+d] = float32(math.Sin(float64(p) * inv))
		}
	}
	embed, err := st.BF16("model.embed_tokens.weight", c.vocab, c.hidden)
	if err != nil {
		return nil, err
	}
	cal.h = make([]float32, cal.rows*c.hidden)
	r := 0
	for _, s := range seqs {
		start := r
		for p, id := range s {
			row := cal.h[r*c.hidden : (r+1)*c.hidden]
			for i, b := range embed[id*c.hidden : (id+1)*c.hidden] {
				row[i] = q8gemm.BF16ToF32(b)
			}
			cal.hidden.apply(row)
			cal.pos = append(cal.pos, p)
			cal.start = append(cal.start, start)
			r++
		}
	}
	return cal, nil
}

// weights reads projection p of a layer and transforms it into the GPU basis.
func (cal *calibration) weights(st *safetensors.Checkpoint, layer, p int, attnNorm, mlpNorm []float32) ([]float32, error) {
	c := cal.c
	n, k := projectionShape(c, p)
	t, err := st.Lookup(fmt.Sprintf("model.layers.%d.%s.weight", layer, gptqProjections[p]), n, k)
	if err != nil {
		return nil, err
	}
	if t.DType != "BF16" {
		return nil, fmt.Errorf("qwen3: %s is %s; the loader expects the official BF16 checkpoint", gptqProjections[p], t.DType)
	}
	raw := make([]uint16, n*k)
	if err := t.ReadBits(raw, 0); err != nil {
		return nil, err
	}
	mat := make([]float32, n*k)
	for i, b := range raw {
		mat[i] = q8gemm.BF16ToF32(b)
	}
	var norm []float32
	in, out := cal.hidden, (*rotation)(nil)
	switch p {
	case 0, 1:
		norm = attnNorm
	case 2:
		norm, out = attnNorm, cal.head
	case 3:
		in, out = cal.head, cal.hidden
	case 4, 5:
		norm = mlpNorm
	case 6:
		in, out = cal.inter, cal.hidden
	}
	parallelRows(n, 64, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := mat[r*k : (r+1)*k]
			if norm != nil {
				for i := range row {
					row[i] *= norm[i]
				}
			}
			in.apply(row)
		}
	})
	if out != nil {
		out.applyRows(mat, n, k)
	}
	return mat, nil
}

// layer rounds one layer's projections with GPTQ, writes them to w in file
// order, and advances the residual stream through the rounded layer.
func (cal *calibration) layer(st *safetensors.Checkpoint, layer, bits int, w io.Writer) error {
	c := cal.c
	p := fmt.Sprintf("model.layers.%d.", layer)
	attnNorm, err := st.Float32(p+"input_layernorm.weight", c.hidden)
	if err != nil {
		return err
	}
	mlpNorm, err := st.Float32(p+"post_attention_layernorm.weight", c.hidden)
	if err != nil {
		return err
	}
	qNorm, err := st.Float32(p+"self_attn.q_norm.weight", c.headDim)
	if err != nil {
		return err
	}
	kNorm, err := st.Float32(p+"self_attn.k_norm.weight", c.headDim)
	if err != nil {
		return err
	}
	var mats [len(gptqProjections)][]float32
	for i := range mats {
		if mats[i], err = cal.weights(st, layer, i, attnNorm, mlpNorm); err != nil {
			return err
		}
	}
	rows, qdim := cal.rows, c.heads*c.headDim
	// Attention: inputs are the normalized residual rows.
	x := normalizedRows(cal.h, rows, c.hidden, c.eps)
	if err := quantizeGroup(x, rows, c.hidden, bits, mats[0:3], w); err != nil {
		return err
	}
	q, err := project(x, rows, c.hidden, mats[0], qdim)
	if err != nil {
		return err
	}
	kk, err := project(x, rows, c.hidden, mats[1], c.kvDim)
	if err != nil {
		return err
	}
	v, err := project(x, rows, c.hidden, mats[2], c.kvDim)
	if err != nil {
		return err
	}
	ctx := cal.attention(q, kk, v, qNorm, kNorm)
	if err := quantizeGroup(ctx, rows, qdim, bits, mats[3:4], w); err != nil {
		return err
	}
	o, err := project(ctx, rows, qdim, mats[3], c.hidden)
	if err != nil {
		return err
	}
	for i := range cal.h {
		cal.h[i] += o[i]
	}
	// MLP.
	x = normalizedRows(cal.h, rows, c.hidden, c.eps)
	if err := quantizeGroup(x, rows, c.hidden, bits, mats[4:6], w); err != nil {
		return err
	}
	gate, err := project(x, rows, c.hidden, mats[4], c.intermediate)
	if err != nil {
		return err
	}
	up, err := project(x, rows, c.hidden, mats[5], c.intermediate)
	if err != nil {
		return err
	}
	parallelRows(rows, 16, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			g, u := gate[r*c.intermediate:(r+1)*c.intermediate], up[r*c.intermediate:(r+1)*c.intermediate]
			for i := range g {
				g[i] = silu32(g[i]) * u[i]
			}
			cal.inter.apply(g)
		}
	})
	if err := quantizeGroup(gate, rows, c.intermediate, bits, mats[6:7], w); err != nil {
		return err
	}
	down, err := project(gate, rows, c.intermediate, mats[6], c.hidden)
	if err != nil {
		return err
	}
	for i := range cal.h {
		cal.h[i] += down[i]
	}
	return nil
}

// attention applies QK-norm and RoPE and attends each row causally within
// its sequence; values keep their per-head rotation, so the context does
// too.
func (cal *calibration) attention(q, k, v, qNorm, kNorm []float32) []float32 {
	c := cal.c
	hd, half, group := c.headDim, c.headDim/2, c.heads/c.kvHeads
	qdim := c.heads * hd
	parallelRows(cal.rows, 16, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := cal.pos[r] * half
			cos, sin := cal.cos[base:base+half], cal.sin[base:base+half]
			for h := range c.heads {
				x := q[r*qdim+h*hd : r*qdim+(h+1)*hd]
				rmsNorm32(x, x, qNorm, c.eps)
				rotateHalves(x[:half], x[half:], cos, sin)
			}
			for h := range c.kvHeads {
				x := k[r*c.kvDim+h*hd : r*c.kvDim+(h+1)*hd]
				rmsNorm32(x, x, kNorm, c.eps)
				rotateHalves(x[:half], x[half:], cos, sin)
			}
		}
	})
	ctx := make([]float32, cal.rows*qdim)
	scale := float32(c.attnScale)
	parallelRows(cal.rows*c.heads, 32, func(lo, hi int) {
		scores := make([]float32, gptqMaxSeq)
		for item := lo; item < hi; item++ {
			r, h := item/c.heads, item%c.heads
			g := h / group
			qr := q[r*qdim+h*hd : r*qdim+(h+1)*hd]
			s := scores[:r-cal.start[r]+1]
			for j := range s {
				kr := k[(cal.start[r]+j)*c.kvDim+g*hd:]
				s[j] = dot32(qr, kr[:hd])
			}
			softmaxScaled(s, scale)
			out := ctx[r*qdim+h*hd : r*qdim+(h+1)*hd]
			for j, p := range s {
				vr := v[(cal.start[r]+j)*c.kvDim+g*hd:]
				axpy32(out, vr[:hd], p)
			}
		}
	})
	return ctx
}

// normalizedRows returns x/rms(x) for each row: RMSNorm without its weight,
// which the next projections hold.
func normalizedRows(x []float32, rows, width int, eps float64) []float32 {
	out := make([]float32, rows*width)
	parallelRows(rows, 64, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			src := x[r*width : (r+1)*width]
			inv := float32(1 / math.Sqrt(float64(sumSquares(src))/float64(width)+eps))
			for i, v := range src {
				out[r*width+i] = v * inv
			}
		}
	})
	return out
}

// project returns x[rows×k]·Wᵀ for W [n][k].
func project(x []float32, rows, k int, w []float32, n int) ([]float32, error) {
	pb, err := whispergemm.NewPackedB(k, n)
	if err != nil {
		return nil, err
	}
	if err := pb.Pack(w, k, true); err != nil {
		return nil, err
	}
	out := make([]float32, rows*n)
	return out, mulInto(out, n, x, k, rows, pb)
}

// quantizeGroup rounds the projections that share inputs x [rows][k] with
// GPTQ, replaces each matrix by its rounded values, and writes codes and
// scales to w.
func quantizeGroup(x []float32, rows, k, bits int, mats [][]float32, w io.Writer) error {
	// H = 2·XᵀX/rows; the factor cancels in GPTQ and is kept for damping scale.
	xt := make([]float32, k*rows)
	parallelRows(rows, 64, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for i, v := range x[r*k : (r+1)*k] {
				xt[i*rows+r] = v
			}
		}
	})
	pb, err := whispergemm.NewPackedB(rows, k)
	if err != nil {
		return err
	}
	if err := pb.Pack(x, k, false); err != nil {
		return err
	}
	h := make([]float32, k*k)
	if err := mulInto(h, k, xt, rows, k, pb); err != nil {
		return err
	}
	var mean float64
	for i := range k {
		mean += float64(h[i*k+i])
	}
	mean /= float64(k)
	dead := make([]bool, k)
	for i := range k {
		if h[i*k+i] == 0 {
			dead[i] = true
			h[i*k+i] = 1
		}
		h[i*k+i] += float32(gptqDamp * mean)
	}
	u, err := gptqInverseFactor(h, k)
	if err != nil {
		return err
	}
	for _, m := range mats {
		n := len(m) / k
		for r := range n {
			for i, d := range dead {
				if d {
					m[r*k+i] = 0
				}
			}
		}
		cb, sb := gptqRowBytes(bits, k)
		codes, scales := make([]byte, n*cb), make([]byte, n*sb)
		if err := gptqRound(m, n, k, u, bits, codes, scales); err != nil {
			return err
		}
		if _, err := w.Write(codes); err != nil {
			return err
		}
		if _, err := w.Write(scales); err != nil {
			return err
		}
	}
	return nil
}

// gptqRound rounds the n×k matrix m in place with GPTQ against U, the upper
// Cholesky factor of H⁻¹, writing codes and scales in the GPU layouts: int8
// rows with one FP32 scale each, or 4-bit blocks of 32 (value j in the low
// nibble of byte j, value j+16 in the high nibble, stored plus 8) with one
// FP16 scale each.
func gptqRound(m []float32, n, k int, u []float32, bits int, codes, scales []byte) error {
	cb, sb := gptqRowBytes(bits, k)
	rowScale := make([]float32, n)
	if bits == 8 {
		for r := range n {
			rowScale[r] = q8gemm.MaxAbs(m[r*k:(r+1)*k]) / 127
			binary.LittleEndian.PutUint32(scales[r*sb:], math.Float32bits(rowScale[r]))
		}
	}
	var mu sync.Mutex
	var first error
	for i1 := 0; i1 < k; i1 += gptqBlock {
		i2 := min(i1+gptqBlock, k)
		b := i2 - i1
		var pb *whispergemm.PackedB
		if i2 < k {
			var err error
			if pb, err = whispergemm.NewPackedB(b, k-i2); err != nil {
				return err
			}
			if err := pb.Pack(u[i1*k+i2:], k, false); err != nil {
				return err
			}
		}
		parallelRows(n, 32, func(lo, hi int) {
			errs := make([]float32, (hi-lo)*b)
			for r := lo; r < hi; r++ {
				row := m[r*k : (r+1)*k]
				s := rowScale[r]
				for i := range b {
					col := i1 + i
					if bits == 4 && col%q4GroupSize == 0 {
						s = q4GroupScale(row[col : col+q4GroupSize])
						binary.LittleEndian.PutUint16(scales[r*sb+2*(col/q4GroupSize):], q8gemm.F32ToF16(s))
					}
					var qv float32
					if bits == 8 {
						c := float32(0)
						if s != 0 {
							c = float32(max(-127, min(127, math.RoundToEven(float64(row[col]/s)))))
						}
						codes[r*cb+col] = byte(int8(c))
						qv = c * s
					} else {
						c := float32(0)
						if s != 0 {
							c = float32(max(-8, min(7, math.RoundToEven(float64(row[col]/s)))))
						}
						at := r*cb + (col/q4GroupSize)*16 + col%16
						nib := byte(int(c) + 8)
						if col%q4GroupSize < 16 {
							codes[at] = codes[at]&0xF0 | nib
						} else {
							codes[at] = codes[at]&0x0F | nib<<4
						}
						qv = c * s
					}
					e := (row[col] - qv) / u[col*k+col]
					row[col] = qv
					urow := u[col*k:]
					for j := i + 1; j < b; j++ {
						row[i1+j] -= e * urow[i1+j]
					}
					errs[(r-lo)*b+i] = e
				}
			}
			if pb == nil {
				return
			}
			upd := make([]float32, (hi-lo)*(k-i2))
			if err := pb.Mul(upd, k-i2, errs, b, hi-lo); err != nil {
				mu.Lock()
				first = errors.Join(first, err)
				mu.Unlock()
				return
			}
			for r := lo; r < hi; r++ {
				row := m[r*k+i2 : (r+1)*k]
				for j, d := range upd[(r-lo)*(k-i2) : (r-lo+1)*(k-i2)] {
					row[j] -= d
				}
			}
		})
		if first != nil {
			return first
		}
	}
	return nil
}

// q4GroupScale returns the FP16-representable scale that minimizes the
// squared rounding error of one block, searching down from max/-8.
func q4GroupScale(v []float32) float32 {
	var peak float32
	for _, x := range v {
		if math.Abs(float64(x)) > math.Abs(float64(peak)) {
			peak = x
		}
	}
	if peak == 0 {
		return 0
	}
	best, bestErr := float32(0), math.Inf(1)
	for step := range 16 {
		d := safetensors.F16ToF32(q8gemm.F32ToF16(peak / -8 * (1 - 0.02*float32(step))))
		if d == 0 {
			continue
		}
		var e float64
		for _, x := range v {
			q := max(-8, min(7, math.RoundToEven(float64(x/d))))
			diff := float64(x) - q*float64(d)
			e += diff * diff
		}
		if e < bestErr {
			best, bestErr = d, e
		}
	}
	return best
}
