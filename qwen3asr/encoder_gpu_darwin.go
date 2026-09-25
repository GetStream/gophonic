// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3asr

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
	"github.com/GetStream/gophonic/internal/nn"
	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/safetensors"
)

//go:embed encoder.metal
var encoderSource string

// convBatch bounds the chunks whose first two convolution outputs are held
// at once (about 6 MiB per chunk).
const convBatch = 16

// gpuMatrix is a linear layer in GPU memory: FP16 rows, each scaled into
// FP16 range by a power of two so every BF16 weight is held exactly, with
// the inverse scales and an FP32 bias.
type gpuMatrix struct {
	w, scale, bias *metal.Buffer
	n, k           int
}

type gpuLayer struct {
	attnW, attnB, ffnW, ffnB *metal.Buffer
	qkv, out, fc1, fc2       gpuMatrix
}

// gpuEncoder is the audio encoder resident in GPU memory.
type gpuEncoder struct {
	e                            *encoder
	dev                          *metal.Device
	gemm, finish                 [4]*metal.Pipeline // store, bias, bias+GELU, residual
	convGELU, conv1, gather      *metal.Pipeline
	layerNorm, attend            *metal.Pipeline
	conv1W, conv1B, postW, postB *metal.Buffer
	positions, zero, gelu        *metal.Buffer
	conv                         [2]gpuMatrix
	convOut, proj1, proj2        gpuMatrix
	layers                       []gpuLayer
	buffers                      []*metal.Buffer
	convArgs                     convArgs
}

type gemmArgs struct{ m, n, k, splitK, splits, padN uint32 }

type convArgs struct {
	frames, chunkFrames, inTime, inFreq, outTime, outFreq, ch, chunk0 uint32
}

type attnArgs struct {
	rows, window, heads, d uint32
	scale                  float32
}

// loadGPUEncoder reads the encoder's weights into GPU memory. It supports
// 64-wide attention heads and channel counts divisible by 16.
func loadGPUEncoder(st *safetensors.Checkpoint, e *encoder, prefix string) (*gpuEncoder, error) {
	if e.headDim != 64 || e.ch%16 != 0 || e.d%64 != 0 || e.ffn%32 != 0 || (9*e.ch)%32 != 0 {
		return nil, errors.New("qwen3asr: the GPU encoder needs 64-wide heads and 16-aligned channels")
	}
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	g := &gpuEncoder{e: e, dev: dev, layers: make([]gpuLayer, e.layerCount)}
	lib, err := dev.Compile(encoderSource)
	if err != nil {
		return nil, fmt.Errorf("qwen3asr: compile encoder kernels: %w", err)
	}
	for _, p := range []struct {
		dst  **metal.Pipeline
		name string
	}{{&g.gemm[epiStore], "gemm_store"}, {&g.gemm[epiBias], "gemm_bias"}, {&g.gemm[epiBiasGELU], "gemm_bias_gelu"},
		{&g.gemm[epiResidual], "gemm_residual"}, {&g.convGELU, "conv_gelu"}, {&g.conv1, "conv1"},
		{&g.finish[epiStore], "gemm_finish_store"}, {&g.finish[epiBias], "gemm_finish_bias"},
		{&g.finish[epiBiasGELU], "gemm_finish_bias_gelu"}, {&g.finish[epiResidual], "gemm_finish_residual"},
		{&g.gather, "gather"}, {&g.layerNorm, "layerNorm"}, {&g.attend, "attend"}} {
		if *p.dst, err = dev.Pipeline(lib, p.name); err != nil {
			return nil, err
		}
	}
	buffer := func(n int) (*metal.Buffer, error) {
		b, err := dev.Buffer(n)
		if err == nil {
			g.buffers = append(g.buffers, b)
		}
		return b, err
	}
	floatsBuffer := func(v []float32) (*metal.Buffer, error) {
		b, err := buffer(4 * len(v))
		if err == nil {
			copy(floats(b.Bytes()), v)
		}
		return b, err
	}
	if g.positions, err = floatsBuffer(e.positions); err != nil {
		return nil, err
	}
	if g.zero, err = buffer(4 * max(3*e.d, e.ffn, e.out, e.ch)); err != nil {
		return nil, err
	}
	table := nn.GELUTable()
	if g.gelu, err = floatsBuffer(unsafe.Slice(&table[0][0], 2*len(table))); err != nil {
		return nil, err
	}

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
	// Buffers are allocated before the jobs run, so jobs only fill them.
	var sends []func() error
	vector := func(name string, n int, dst **metal.Buffer) error {
		b, err := buffer(4 * n)
		*dst = b
		if err != nil {
			return err
		}
		sends = append(sends, func() error {
			v, err := st.Float32(prefix+name, n)
			copy(floats(b.Bytes()), v)
			return err
		})
		return nil
	}
	// matrix reads [n][k] BF16 rows of each name (stacked), reorders
	// columns with perm when given, and stores them as scaled FP16.
	matrix := func(names []string, n, k int, perm func(row []uint16, tmp []uint16), bias []string, dst *gpuMatrix, shape ...int) error {
		if shape == nil {
			shape = []int{n, k}
		}
		rows := n * len(names)
		*dst = gpuMatrix{n: rows, k: k}
		var err error
		if dst.w, err = buffer(2 * rows * k); err != nil {
			return err
		}
		if dst.scale, err = buffer(4 * rows); err != nil {
			return err
		}
		dst.bias = g.zero
		if bias != nil {
			if dst.bias, err = buffer(4 * rows); err != nil {
				return err
			}
		}
		for i, name := range names {
			sends = append(sends, func() error {
				t, err := st.Lookup(prefix+name, shape...)
				if err != nil {
					return err
				}
				if t.DType != "BF16" {
					return fmt.Errorf("qwen3asr: %s is %s, not the official BF16", name, t.DType)
				}
				raw := make([]uint16, n*k)
				if err := t.ReadBits(raw, 0); err != nil {
					return err
				}
				if perm != nil {
					tmp := make([]uint16, k)
					for r := range n {
						perm(raw[r*k:(r+1)*k], tmp)
					}
				}
				halves := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(dst.w.Bytes()))), rows*k)[i*n*k : (i+1)*n*k]
				halfRows(raw, n, k, halves, floats(dst.scale.Bytes())[i*n:(i+1)*n])
				if bias != nil {
					b, err := st.Float32(prefix+bias[i], n)
					copy(floats(dst.bias.Bytes())[i*n:], b)
					return err
				}
				return nil
			})
		}
		return nil
	}
	// Convolution kernels [out][in][3 freq][3 time] become rows over
	// (time tap, frequency tap, channel), as the implicit im2col reads them.
	convPerm := func(in int) func(row, tmp []uint16) {
		return func(row, tmp []uint16) {
			for c := range in {
				for fi := range 3 {
					for tj := range 3 {
						tmp[(tj*3+fi)*in+c] = row[(c*3+fi)*3+tj]
					}
				}
			}
			copy(row, tmp)
		}
	}
	f3 := e.freq[3]
	outPerm := func(row, tmp []uint16) { // c·F+f to f·C+c
		for c := range e.ch {
			for f := range f3 {
				tmp[f*e.ch+c] = row[c*f3+f]
			}
		}
		copy(row, tmp)
	}
	check := func(errs ...error) error {
		for _, err := range errs {
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := check(
		vector("conv2d1.bias", e.ch, &g.conv1B),
		matrix([]string{"conv2d2.weight"}, e.ch, 9*e.ch, convPerm(e.ch), []string{"conv2d2.bias"}, &g.conv[0], e.ch, e.ch, 3, 3),
		matrix([]string{"conv2d3.weight"}, e.ch, 9*e.ch, convPerm(e.ch), []string{"conv2d3.bias"}, &g.conv[1], e.ch, e.ch, 3, 3),
		matrix([]string{"conv_out.weight"}, e.d, f3*e.ch, outPerm, nil, &g.convOut),
		vector("ln_post.weight", e.d, &g.postW), vector("ln_post.bias", e.d, &g.postB),
		matrix([]string{"proj1.weight"}, e.d, e.d, nil, []string{"proj1.bias"}, &g.proj1),
		matrix([]string{"proj2.weight"}, e.out, e.d, nil, []string{"proj2.bias"}, &g.proj2),
	); err != nil {
		return nil, err
	}
	// conv1 has one input channel; its kernel runs directly in FP32.
	if g.conv1W, err = buffer(4 * 9 * e.ch); err != nil {
		return nil, err
	}
	sends = append(sends, func() error {
		w, err := st.Float32(prefix+"conv2d1.weight", e.ch, 1, 3, 3)
		dst := floats(g.conv1W.Bytes())
		for c := range e.ch {
			for fi := range 3 {
				for tj := range 3 {
					dst[c*9+tj*3+fi] = w[c*9+fi*3+tj]
				}
			}
		}
		return err
	})
	for i := range g.layers {
		l, p := &g.layers[i], fmt.Sprintf("layers.%d.", i)
		if err := check(
			vector(p+"self_attn_layer_norm.weight", e.d, &l.attnW), vector(p+"self_attn_layer_norm.bias", e.d, &l.attnB),
			vector(p+"final_layer_norm.weight", e.d, &l.ffnW), vector(p+"final_layer_norm.bias", e.d, &l.ffnB),
			matrix([]string{p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"}, e.d, e.d, nil,
				[]string{p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias", p + "self_attn.v_proj.bias"}, &l.qkv),
			matrix([]string{p + "self_attn.out_proj.weight"}, e.d, e.d, nil, []string{p + "self_attn.out_proj.bias"}, &l.out),
			matrix([]string{p + "fc1.weight"}, e.ffn, e.d, nil, []string{p + "fc1.bias"}, &l.fc1),
			matrix([]string{p + "fc2.weight"}, e.d, e.ffn, nil, []string{p + "fc2.bias"}, &l.fc2),
		); err != nil {
			return nil, err
		}
	}
	for _, job := range sends {
		jobs <- job
	}
	close(jobs)
	wg.Wait()
	if first != nil {
		g.release()
		return nil, fmt.Errorf("qwen3asr: audio encoder: %w", first)
	}
	return g, nil
}

// halfRows stores each BF16 row times the power of two that puts its
// largest magnitude in [2^14, 2^15) as FP16, which is exact for every value
// within 2^-24 of that magnitude, and the inverse power as the row's scale.
func halfRows(bf16 []uint16, n, k int, dst []uint16, scale []float32) {
	for r := range n {
		row := bf16[r*k : (r+1)*k]
		var peak float32
		for _, b := range row {
			peak = max(peak, float32(math.Abs(float64(q8gemm.BF16ToF32(b)))))
		}
		e := 0
		if peak > 0 {
			_, exp := math.Frexp(float64(peak))
			e = min(max(15-exp, -100), 100)
		}
		s := float32(math.Ldexp(1, e))
		scale[r] = float32(math.Ldexp(1, -e))
		for i, b := range row {
			dst[r*k+i] = q8gemm.F32ToF16(q8gemm.BF16ToF32(b) * s)
		}
	}
}

func floats(b []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/4)
}

func (g *gpuEncoder) release() {
	for _, b := range g.buffers {
		b.Release()
	}
	g.buffers = nil
	g.dev.Close()
}

// gpuEncoderWorkspace holds one lane's encoder activations in GPU memory.
// Buffers grow to the longest audio seen.
type gpuEncoderWorkspace struct {
	g                           *gpuEncoder
	enc                         metal.Encoder
	mel, c1, c2, stacked, co    *metal.Buffer
	x, norm, qkv, ctx, ffn, out *metal.Buffer
	rowMap, scratch             *metal.Buffer
	gemm                        gemmArgs
	conv                        convArgs
	attn                        attnArgs
	width                       uint32
}

func (g *gpuEncoder) newWorkspace() *gpuEncoderWorkspace { return &gpuEncoderWorkspace{g: g} }

func (w *gpuEncoderWorkspace) close() {
	for _, b := range []*metal.Buffer{w.mel, w.c1, w.c2, w.stacked, w.co, w.x, w.norm, w.qkv, w.ctx, w.ffn, w.out, w.rowMap, w.scratch} {
		b.Release()
	}
}

// fit returns b if it holds n bytes, and otherwise a new buffer.
func (w *gpuEncoderWorkspace) fit(b **metal.Buffer, n int) error {
	if *b != nil && len((*b).Bytes()) >= n {
		return nil
	}
	nb, err := w.g.dev.Buffer(n)
	if err != nil {
		return err
	}
	(*b).Release()
	*b = nb
	return nil
}

// features returns GPU-visible storage for n feature values, which the
// frontend fills before encode.
func (w *gpuEncoderWorkspace) features(n int) ([]float32, error) {
	if err := w.fit(&w.mel, 4*n); err != nil {
		return nil, err
	}
	return floats(w.mel.Bytes())[:n], nil
}

// encode runs the encoder on the [bins][frames] features in w.mel and
// returns the [tokens][out] embeddings, which stay valid until the next
// call.
func (w *gpuEncoderWorkspace) encode(frames int) ([]float32, error) {
	g, e := w.g, w.g.e
	chunks := (frames + e.chunkFrames - 1) / e.chunkFrames
	span := e.chunkFrames
	if chunks == 1 {
		span = frames
	}
	t1, t2, t3 := convLen(span), convLen(convLen(span)), frameTokens(span)
	f1, f2, f3 := e.freq[1], e.freq[2], e.freq[3]
	n, d := e.tokens(frames), e.d
	batch := min(chunks, convBatch)
	for _, b := range []struct {
		dst **metal.Buffer
		n   int
	}{
		{&w.c1, batch * t1 * f1 * e.ch}, {&w.c2, batch * t2 * f2 * e.ch}, {&w.stacked, chunks * t3 * f3 * e.ch},
		{&w.co, chunks * t3 * d}, {&w.x, n * d}, {&w.norm, n * d}, {&w.qkv, 3 * n * d}, {&w.ctx, n * d},
		{&w.ffn, n * max(e.ffn, d)}, {&w.out, n * e.out}, {&w.rowMap, 2 * n},
		{&w.scratch, max(gemmScratch(batch*t2*f2, e.ch, 9*e.ch), gemmScratch(batch*t3*f3, e.ch, 9*e.ch),
			gemmScratch(chunks*t3, d, f3*e.ch), gemmScratch(n, 3*d, d), gemmScratch(n, e.ffn, d),
			gemmScratch(n, d, e.ffn), gemmScratch(n, e.out, d))},
	} {
		if err := w.fit(b.dst, 4*b.n); err != nil {
			return nil, err
		}
	}
	rowMap := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(w.rowMap.Bytes()))), 2*n)
	i := 0
	for c := range chunks {
		for t := range frameTokens(min(e.chunkFrames, frames-c*e.chunkFrames)) {
			rowMap[2*i], rowMap[2*i+1] = uint32(c*t3+t), uint32(t)
			i++
		}
	}
	enc := &w.enc
	g.dev.Begin(enc, false)
	for b0 := 0; b0 < chunks; b0 += batch {
		nb := min(batch, chunks-b0)
		w.conv = convArgs{frames: uint32(frames), chunkFrames: uint32(e.chunkFrames), inFreq: uint32(e.freq[0]),
			outTime: uint32(t1), outFreq: uint32(f1), ch: uint32(e.ch), chunk0: uint32(b0)}
		enc.SetPipeline(g.conv1)
		enc.SetBuffer(w.mel, 0, 0)
		enc.SetBuffer(g.conv1W, 0, 1)
		enc.SetBuffer(g.conv1B, 0, 2)
		enc.SetBuffer(w.c1, 0, 3)
		enc.SetBytes(unsafe.Pointer(&w.conv), int(unsafe.Sizeof(w.conv)), 4)
		enc.SetBuffer(g.gelu, 0, 5)
		enc.Dispatch(metal.Size{X: (e.ch + 63) / 64, Y: t1 * f1, Z: nb}, metal.Size{X: 64, Y: 1, Z: 1})
		w.conv = convArgs{inTime: uint32(t1), inFreq: uint32(f1), outTime: uint32(t2), outFreq: uint32(f2), ch: uint32(e.ch)}
		w.dispatchGEMM(g.convGELU, epiBiasGELU, &g.conv[0], w.c1, 0, w.c2, 0, nb*t2*f2)
		w.conv = convArgs{inTime: uint32(t2), inFreq: uint32(f2), outTime: uint32(t3), outFreq: uint32(f3), ch: uint32(e.ch)}
		w.dispatchGEMM(g.convGELU, epiBiasGELU, &g.conv[1], w.c2, 0, w.stacked, 4*b0*t3*f3*e.ch, nb*t3*f3)
	}
	w.dispatchGEMM(g.gemm[epiStore], epiStore, &g.convOut, w.stacked, 0, w.co, 0, chunks*t3)
	w.width = uint32(d)
	enc.SetPipeline(g.gather)
	enc.SetBuffer(w.co, 0, 0)
	enc.SetBuffer(w.x, 0, 1)
	enc.SetBuffer(w.rowMap, 0, 2)
	enc.SetBuffer(g.positions, 0, 3)
	enc.SetBytes(unsafe.Pointer(&w.width), 4, 4)
	enc.Dispatch(metal.Size{X: (d/4 + 63) / 64, Y: n, Z: 1}, metal.Size{X: 64, Y: 1, Z: 1})

	w.norm2(w.x, w.norm, g.layers[0].attnW, g.layers[0].attnB, n)
	window := frameTokens(span) * e.windowChunks
	w.attn = attnArgs{rows: uint32(n), window: uint32(window), heads: uint32(e.heads), d: uint32(d), scale: float32(1 / math.Sqrt(float64(e.headDim)))}
	for li := range g.layers {
		l := &g.layers[li]
		nextW, nextB := g.postW, g.postB
		if li+1 < len(g.layers) {
			nextW, nextB = g.layers[li+1].attnW, g.layers[li+1].attnB
		}
		w.dispatchGEMM(g.gemm[epiBias], epiBias, &l.qkv, w.norm, 0, w.qkv, 0, n)
		enc.SetPipeline(g.attend)
		enc.SetBuffer(w.qkv, 0, 0)
		enc.SetBuffer(w.ctx, 0, 1)
		enc.SetBytes(unsafe.Pointer(&w.attn), int(unsafe.Sizeof(w.attn)), 2)
		enc.Dispatch(metal.Size{X: e.heads, Y: (n + 3) / 4, Z: 1}, metal.Size{X: 128, Y: 1, Z: 1})
		w.dispatchGEMM(g.gemm[epiResidual], epiResidual, &l.out, w.ctx, 0, w.x, 0, n)
		w.norm2(w.x, w.norm, l.ffnW, l.ffnB, n)
		w.dispatchGEMM(g.gemm[epiBiasGELU], epiBiasGELU, &l.fc1, w.norm, 0, w.ffn, 0, n)
		w.dispatchGEMM(g.gemm[epiResidual], epiResidual, &l.fc2, w.ffn, 0, w.x, 0, n)
		w.norm2(w.x, w.norm, nextW, nextB, n)
	}
	w.dispatchGEMM(g.gemm[epiBiasGELU], epiBiasGELU, &g.proj1, w.norm, 0, w.ffn, 0, n)
	w.dispatchGEMM(g.gemm[epiBias], epiBias, &g.proj2, w.ffn, 0, w.out, 0, n)
	if err := enc.Wait(); err != nil {
		return nil, err
	}
	return floats(w.out.Bytes())[:n*e.out], nil
}

// gemmTile and gemmThreads are the GEMM kernel's output tile (BM = BN) and
// threadgroup size in encoder.metal; gemmBK is its K step.
const gemmTile, gemmThreads, gemmBK = 64, 128, 32

// gemmMinGroups is the threadgroup count below which a product splits K,
// so that every GPU core has several threadgroups to switch between.
const gemmMinGroups = 128

// gemmSplits returns how many ways to split K for an m×n product.
func gemmSplits(m, n, k int) int {
	groups := (n + gemmTile - 1) / gemmTile * ((m + gemmTile - 1) / gemmTile)
	splits := 1
	for splits < 8 && groups*splits < gemmMinGroups && k%(2*splits*gemmBK) == 0 {
		splits *= 2
	}
	return splits
}

// gemmScratch is the split-K scratch, in floats, of an m×n product over k.
func gemmScratch(m, n, k int) int {
	splits := gemmSplits(m, n, k)
	if splits == 1 {
		return 0
	}
	return splits * (m + gemmTile) * ((n + gemmTile - 1) / gemmTile * gemmTile)
}

// dispatchGEMM encodes y = x·Wᵀ with epilogue epi (pipeline p) over m rows;
// x and y start at byte offsets xOff and yOff. Convolution kernels read
// w.conv.
func (w *gpuEncoderWorkspace) dispatchGEMM(p *metal.Pipeline, epi int, mat *gpuMatrix, x *metal.Buffer, xOff int, y *metal.Buffer, yOff, m int) {
	enc := &w.enc
	splits := gemmSplits(m, mat.n, mat.k)
	padN := (mat.n + gemmTile - 1) / gemmTile * gemmTile
	w.gemm = gemmArgs{m: uint32(m), n: uint32(mat.n), k: uint32(mat.k), splitK: uint32(mat.k / splits),
		splits: uint32(splits), padN: uint32(padN)}
	enc.SetPipeline(p)
	enc.SetBuffer(x, xOff, 0)
	enc.SetBuffer(mat.w, 0, 1)
	enc.SetBuffer(mat.scale, 0, 2)
	enc.SetBuffer(mat.bias, 0, 3)
	enc.SetBuffer(y, yOff, 4)
	enc.SetBytes(unsafe.Pointer(&w.gemm), int(unsafe.Sizeof(w.gemm)), 5)
	enc.SetBytes(unsafe.Pointer(&w.conv), int(unsafe.Sizeof(w.conv)), 6)
	enc.SetBuffer(w.g.gelu, 0, 7)
	enc.SetBuffer(w.scratch, 0, 8)
	enc.Dispatch(metal.Size{X: padN / gemmTile, Y: (m + gemmTile - 1) / gemmTile, Z: splits}, metal.Size{X: gemmThreads, Y: 1, Z: 1})
	if splits > 1 {
		enc.SetPipeline(w.g.finish[epi])
		enc.Dispatch(metal.Size{X: padN / 64, Y: m, Z: 1}, metal.Size{X: 64, Y: 1, Z: 1})
	}
}

// norm2 encodes y = LayerNorm(x)·w + b over n rows.
func (w *gpuEncoderWorkspace) norm2(x, y, weight, bias *metal.Buffer, n int) {
	enc := &w.enc
	w.width = uint32(w.g.e.d)
	enc.SetPipeline(w.g.layerNorm)
	enc.SetBuffer(x, 0, 0)
	enc.SetBuffer(y, 0, 1)
	enc.SetBuffer(weight, 0, 2)
	enc.SetBuffer(bias, 0, 3)
	enc.SetBytes(unsafe.Pointer(&w.width), 4, 4)
	enc.Dispatch(metal.Size{X: n, Y: 1, Z: 1}, metal.Size{X: 256, Y: 1, Z: 1})
}
