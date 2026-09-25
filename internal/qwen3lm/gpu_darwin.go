// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/safetensors"
)

//go:embed gpu.metal
var gpuSource string

const (
	gpuThreads = 256
	gpuAlign   = 256
	// gpuPositions is the rows of one GPU pass and the positions of a GPU
	// workspace's key and value cache.
	gpuPositions = 2048
	// gpuMaxPositions bounds the RoPE table, and so the positions a GPU
	// prefix can hold.
	gpuMaxPositions = 1 << 16
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
	rotate, qkRope, attendM      *metal.Pipeline
	attendFlash, gemvHead        *metal.Pipeline
	mm                           [2][4]*metal.Pipeline // [32-token, 16-token tiles][qkv, o, gateup, down]
	finish                       [3]*metal.Pipeline    // split-K epilogues: store, add, SwiGLU
	layers                       []gpuLayer
	norms, signs, rope           *metal.Buffer
	hidden, inter, head          *rotation
	cfg                          *modelConfig
	noInter                      bool // down's inputs are not rotated
	bits                         int  // weight scheme, as in gpu.metal: 8 (Q8), 9 (Q8B), or 4 (Q4)
	positions                    int  // RoPE table rows
	// lm is the language-model head W·Rᵀ as Q8B int8 blocks with FP16
	// scales, padded with zero rows to lmRows; nil unless loaded.
	lm      *metal.Buffer
	lmScale int // byte offset of the scales in lm
	lmRows  int
}

// gpuRows is the number of weight rows per GEMV threadgroup: 8 simdgroups
// times rowsPerSimdgroup in gpu.metal.
func gpuRows(bits int) int {
	if bits != 4 {
		return 16
	}
	return 32
}

func alignUp(n int) int { return (n + gpuAlign - 1) &^ (gpuAlign - 1) }

// gpuGeometry checks that the GPU kernels, which gpu.metal specializes by
// the model's widths, fit the model: 128-wide heads, at most eight query
// heads per key/value head, and projection widths that tile evenly.
func gpuGeometry(c *modelConfig) error {
	qdim := c.heads * c.headDim
	group := c.heads / c.kvHeads
	rot := newRotation(c.intermediate).block
	if c.headDim != 128 || group > 8 || c.hidden%mmColumns != 0 || c.intermediate%mmColumns != 0 ||
		(qdim+2*c.kvDim)%mmColumns != 0 || rot < 64 {
		return errors.New("qwen3: the GPU backend needs 128-wide heads, at most 8 query heads per key/value head, and 64-aligned widths")
	}
	return nil
}

// gpuSourceFor returns gpu.metal specialized for the model's geometry.
func gpuSourceFor(c *modelConfig) string {
	return fmt.Sprintf("#define QD %d\n#define KVD %d\n#define NH %d\n#define NKV %d\n#define ROT %d\n",
		c.heads*c.headDim, c.kvDim, c.heads, c.kvHeads, newRotation(c.intermediate).block) + gpuSource
}

// gpuSupports reports whether a Metal GPU is present and the model has a
// geometry the GPU kernels support.
func gpuSupports(c *modelConfig) bool {
	return gpuGeometry(c) == nil && GPUAvailable()
}

var (
	gpuProbe   sync.Once
	gpuPresent bool
)

// loadGPU quantizes every projection, and the head when named, into GPU
// buffers.
func (m *Weights) loadGPU(st *safetensors.Checkpoint, bits int, headName string) error {
	c := &m.cfg
	if err := gpuGeometry(c); err != nil {
		return err
	}
	dev, err := metal.Open()
	if err != nil {
		return err
	}
	g := &gpuModel{dev: dev, cfg: c, layers: make([]gpuLayer, c.layers), bits: bits,
		positions: min(c.maxPositions, gpuMaxPositions)}
	suffix := map[int]string{8: "", 9: "_q8", 4: "_q4"}[bits]
	lib, err := dev.Compile(gpuSourceFor(c))
	if err != nil {
		return err
	}
	for _, p := range []struct {
		dst  **metal.Pipeline
		name string
	}{{&g.qkv, "gemv_qkv" + suffix}, {&g.o, "gemv_o" + suffix}, {&g.gateup, "gemv_gateup" + suffix}, {&g.down, "gemv_down" + suffix}, {&g.attend, "attend1"}, {&g.rotate, "rotate"},
		{&g.gemvHead, "gemv_head"},
		{&g.qkRope, "qkRope"}, {&g.attendM, "attendM"}, {&g.attendFlash, "attendFlash"},
		{&g.mm[0][0], "mm_qkv" + suffix}, {&g.mm[0][1], "mm_o" + suffix}, {&g.mm[0][2], "mm_gateup" + suffix}, {&g.mm[0][3], "mm_down" + suffix},
		{&g.finish[0], "mm_finish_store" + suffix}, {&g.finish[1], "mm_finish_add" + suffix}, {&g.finish[2], "mm_finish_swiglu" + suffix},
		{&g.mm[1][0], "mm_qkv" + suffix + "_16"}, {&g.mm[1][1], "mm_o" + suffix + "_16"}, {&g.mm[1][2], "mm_gateup" + suffix + "_16"}, {&g.mm[1][3], "mm_down" + suffix + "_16"}} {
		if *p.dst, err = dev.Pipeline(lib, p.name); err != nil {
			return err
		}
	}
	h, kv, inter, qdim := c.hidden, c.kvDim, c.intermediate, c.heads*c.headDim
	g.hidden, g.inter = newRotation(h), newRotation(inter)
	// Block scales absorb the outliers of down's input that the online
	// Hadamard otherwise spreads, at the same fidelity, so Q8B skips that
	// dispatch per layer.
	downIn := g.inter
	if bits == 9 {
		downIn, g.noInter = nil, true
	}
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
	if g.rope, err = dev.Buffer(4 * 2 * g.positions * half); err != nil {
		return err
	}
	rope := floats(g.rope.Bytes())
	for pos := range g.positions {
		for d, inv := range c.invFreq {
			theta := float64(pos) * inv
			rope[pos*half+d] = float32(math.Cos(theta))
			rope[g.positions*half+pos*half+d] = float32(math.Sin(theta))
		}
	}

	type job struct {
		name     string
		n, k     int
		norm     []float32 // folded input RMSNorm weight
		in, out  *rotation // input- and output-side rotations
		buf      *metal.Buffer
		base, sc int // byte offsets of row 0 and scale 0
		step     int // destination row stride in rows
		scaleMul float32
		row0     int    // destination row of source row 0
		rows     [2]int // for a chunk of a large matrix: its source rows
	}
	var jobs []job
	for i := range m.layers {
		l, gl := &m.layers[i], &g.layers[i]
		off := 0
		place := func(rows, k int) (int, int) {
			w := off
			if bits == 4 {
				off = alignUp(off + rows*k/2)
			} else {
				off = alignUp(off + rows*k)
			}
			s := off
			if bits == 8 {
				off = alignUp(off + 4*rows)
			} else {
				off = alignUp(off + 2*rows*(k/q4Group))
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
		p := fmt.Sprintf("%slayers.%d.", m.prefix, i)
		jobs = append(jobs,
			job{p + "self_attn.q_proj.weight", qdim, h, l.attnNorm, g.hidden, nil, gl.buf, gl.qkv, gl.qkvScale, 1, 1, 0, [2]int{}},
			job{p + "self_attn.k_proj.weight", kv, h, l.attnNorm, g.hidden, nil, gl.buf, gl.qkv, gl.qkvScale, 1, 1, qdim, [2]int{}},
			job{p + "self_attn.v_proj.weight", kv, h, l.attnNorm, g.hidden, g.head, gl.buf, gl.qkv, gl.qkvScale, 1, 1, qdim + kv, [2]int{}},
			job{p + "self_attn.o_proj.weight", h, qdim, nil, g.head, g.hidden, gl.buf, gl.o, gl.oScale, 1, 1, 0, [2]int{}},
			job{p + "mlp.gate_proj.weight", inter, h, l.mlpNorm, g.hidden, nil, gl.buf, gl.gu, gl.guScale, 2, 1, 0, [2]int{}},
			job{p + "mlp.up_proj.weight", inter, h, l.mlpNorm, g.hidden, nil, gl.buf, gl.gu, gl.guScale, 2, 1, 1, [2]int{}},
			job{p + "mlp.down_proj.weight", h, inter, nil, downIn, g.hidden, gl.buf, gl.d, gl.dScale, 1, 1, 0, [2]int{}},
		)
	}
	if headName != "" {
		// The head always stores W·Rᵀ as int8 blocks (Q8B): LogitsInto
		// rotates the normalized state. It is read in chunks of at most one
		// projection.
		g.lmRows = (c.vocab + gpuRows(9) - 1) / gpuRows(9) * gpuRows(9)
		g.lmScale = alignUp(g.lmRows * h)
		if g.lm, err = dev.Buffer(g.lmScale + 2*g.lmRows*(h/q4Group)); err != nil {
			return err
		}
		step := inter
		for r := 0; r < c.vocab; r += step {
			jobs = append(jobs, job{headName, c.vocab, h, nil, g.hidden, nil, g.lm, 0, g.lmScale, 1, 1, 0, [2]int{r, min(r+step, c.vocab)}})
		}
	}
	// GPTQ-rounded weights made by QuantizeGPTQ replace round-to-nearest.
	var pre *gptqFile
	if format := map[int]string{8: WeightsGPU, 4: WeightsGPUQ4}[bits]; format != "" {
		if pre = openGPTQ(st.Dir(), format, bits, c); pre != nil {
			defer pre.close()
		}
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
				if pre != nil {
					done, err := pre.place(strings.TrimPrefix(j.name, m.prefix), j.n, j.k, j.buf.Bytes(), j.base, j.sc, j.row0, j.step)
					if err != nil {
						mu.Lock()
						first = errors.Join(first, err)
						mu.Unlock()
						return
					}
					if done {
						continue
					}
				}
				r0, n := 0, j.n
				if j.rows[1] > 0 {
					r0, n = j.rows[0], j.rows[1]-j.rows[0]
				}
				t, err := st.Lookup(j.name, j.n, j.k)
				if err == nil && t.DType != "BF16" {
					err = fmt.Errorf("qwen3: %s is %s; the loader expects the official BF16 checkpoint", j.name, t.DType)
				}
				if err == nil {
					raw = grow(raw, n*j.k)
					err = t.ReadBits(raw, int64(r0)*int64(j.k))
				}
				if err != nil {
					mu.Lock()
					first = errors.Join(first, err)
					mu.Unlock()
					return
				}
				mat = grow(mat, n*j.k)
				for i, b := range raw {
					mat[i] = q8gemm.BF16ToF32(b)
				}
				for r := range n {
					row := mat[r*j.k : (r+1)*j.k]
					if j.norm != nil {
						for i := range row {
							row[i] *= j.norm[i]
						}
					}
					if j.in != nil {
						j.in.apply(row)
					}
				}
				if j.out != nil {
					j.out.applyRows(mat, j.n, j.k)
				}
				buf := j.buf.Bytes()
				for r := range n {
					row := mat[r*j.k : (r+1)*j.k]
					dst := j.row0 + (r0+r)*j.step
					scheme := bits
					if j.buf == g.lm {
						scheme = 9
					}
					groups := j.k / q4Group
					switch scheme {
					case 4:
						sc := unsafe.Slice((*uint16)(unsafe.Pointer(&buf[j.sc+2*dst*groups])), groups)
						quantizeRowQ4(row, buf[j.base+dst*j.k/2:][:j.k/2], sc, j.scaleMul)
					case 9:
						sc := unsafe.Slice((*uint16)(unsafe.Pointer(&buf[j.sc+2*dst*groups])), groups)
						q := unsafe.Slice((*int8)(unsafe.Pointer(&buf[j.base+dst*j.k])), j.k)
						quantizeRowQ8B(row, q, sc, j.scaleMul)
					default:
						q := unsafe.Slice((*int8)(unsafe.Pointer(&buf[j.base+dst*j.k])), j.k)
						floats(buf[j.sc:])[dst] = quantizeRow(row, q) * j.scaleMul
					}
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

// q4Group is the block length along K of the Q4 and Q8B schemes.
const q4Group = 32

// quantizeRowQ8B stores each 32-value block of row as int8 codes with one
// FP16 scale max|v|/127 (times mul), as GGML's Q8_0 does.
func quantizeRowQ8B(row []float32, q []int8, scales []uint16, mul float32) {
	for b := range len(row) / q4Group {
		v := row[b*q4Group : (b+1)*q4Group]
		d := safetensors.F16ToF32(q8gemm.F32ToF16(q8gemm.MaxAbs(v) / 127))
		scales[b] = q8gemm.F32ToF16(d * mul)
		if d == 0 {
			clear(q[b*q4Group : (b+1)*q4Group])
			continue
		}
		for i, x := range v {
			q[b*q4Group+i] = int8(max(-127, min(127, math.RoundToEven(float64(x/d)))))
		}
	}
}

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
				d = safetensors.F16ToF32(q8gemm.F32ToF16(d))
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
	g                    *gpuModel
	h, qkv, ctx, act, kc *metal.Buffer
	vc, embedParts, info *metal.Buffer // info: per row (position, sequence start row)
	scratch              *metal.Buffer // split-K partial sums
	// The current pass's caches: new keys and values go to curK/curV at row
	// attn.base+m (layer stride curStride bytes); preK/preV hold a read-only
	// prefix of attn.prefixLen rows (layer stride preStride).
	curK, curV, preK, preV                  *metal.Buffer
	curStride, preStride                    int
	attnParts, mlpParts                     *metal.Buffer // residual sums of squares for the next RMSNorm
	enc                                     metal.Encoder
	qkv0Args, qkvArgs, oArgs, guArgs, dArgs gemvArgs
	attn                                    attnArgs
	mm                                      mmArgs
	perRow, batchRows                       uint32
	rows                                    int
	past                                    int
	shared                                  bool
	embeds                                  Embeds
	spliced                                 int // embeds rows used so far
	logits                                  *metal.Buffer
	headArgs                                gemvArgs
	oneSeq                                  [1][]int
}

// gpuTokenByToken forces the single-token kernels and gpuScalarAttention
// the per-key attention loop; tests compare the paths.
var gpuTokenByToken, gpuScalarAttention bool

// attendSplits is the simdgroups that share one query head's keys in the
// single-token attention (AS in gpu.metal).
const attendSplits = 4

// mmColumns is the GEMM tile width in weight rows (MM_BN in gpu.metal).
const mmColumns = 64

type mmArgs struct {
	k, n, m         uint32
	eps             float32
	parts, partsOut uint32
	splitK, splits  uint32
	padM            uint32
}

// mmMinGroups is the threadgroup count below which projections split K.
const mmMinGroups = 256

// mmScratchFloats bounds split-K scratch: splits only happen while the grid
// is below mmMinGroups tiles of at most 32×64, and splits are at most 8.
const mmScratchFloats = 8 * mmMinGroups * 32 * mmColumns

type gemvArgs struct {
	k, n  uint32
	eps   float32
	parts uint32
}

type attnArgs struct {
	pos, ropeSin    uint32
	eps, scale      float32
	base, prefixLen uint32
}

// gpuPrefix is a PrefixKV's per-layer keys and values in GPU memory,
// [layers][capacity][kvDim] FP32 each.
type gpuPrefix struct {
	kc, vc   *metal.Buffer
	capacity int
}

func (g *gpuModel) newPrefix(capacity int) (*gpuPrefix, error) {
	n := 4 * g.cfg.layers * capacity * g.cfg.kvDim
	kc, err := g.dev.Buffer(n)
	if err != nil {
		return nil, err
	}
	vc, err := g.dev.Buffer(n)
	if err != nil {
		kc.Release()
		return nil, err
	}
	return &gpuPrefix{kc: kc, vc: vc, capacity: capacity}, nil
}

// copyFrom copies the first n values of each layer's keys and values.
func (p *gpuPrefix) copyFrom(src *gpuPrefix, layers, n int) {
	dk, dv := floats(p.kc.Bytes()), floats(p.vc.Bytes())
	sk, sv := floats(src.kc.Bytes()), floats(src.vc.Bytes())
	dStride, sStride := len(dk)/layers, len(sk)/layers
	for l := range layers {
		copy(dk[l*dStride:l*dStride+n], sk[l*sStride:l*sStride+n])
		copy(dv[l*dStride:l*dStride+n], sv[l*sStride:l*sStride+n])
	}
}

func (p *gpuPrefix) release() {
	p.kc.Release()
	p.vc.Release()
}

func (g *gpuModel) newWorkspace() (*gpuWorkspace, error) {
	c := g.cfg
	qdim := c.heads * c.headDim
	rows := gpuRows(g.bits)
	w := &gpuWorkspace{g: g, rows: gpuPositions}
	var err error
	for _, b := range []struct {
		dst **metal.Buffer
		n   int
	}{
		{&w.h, 4 * gpuPositions * c.hidden},
		{&w.qkv, 4 * gpuPositions * (qdim + 2*c.kvDim)},
		{&w.ctx, 4 * gpuPositions * qdim},
		{&w.act, 4 * gpuPositions * c.intermediate},
		{&w.kc, 4 * c.layers * gpuPositions * c.kvDim},
		{&w.vc, 4 * c.layers * gpuPositions * c.kvDim},
		{&w.embedParts, 4 * gpuPositions},
		{&w.info, 8 * gpuPositions},
		{&w.scratch, 4 * mmScratchFloats},
		{&w.attnParts, 4 * max(c.hidden/rows, gpuPositions*c.hidden/mmColumns)},
		{&w.mlpParts, 4 * max(c.hidden/rows, gpuPositions*c.hidden/mmColumns)},
		{&w.logits, 4 * max(g.lmRows, 1)},
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
	w.headArgs = gemvArgs{uint32(c.hidden), uint32(g.lmRows), eps, 0}
	w.attn = attnArgs{ropeSin: uint32(g.positions * c.headDim / 2), eps: eps, scale: float32(c.attnScale)}
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

// batch evaluates independent sequences and writes each one's last-token
// post-final-norm state to dst. Sequences are packed into shared forward
// passes of up to gpuPositions rows; a lone single token takes the GEMV path.
// With a prefix pre of past tokens, sequences continue it: shared prefixes
// are only read, and an unshared one (a single sequence) receives the new
// keys and values after its first past rows, a pass at a time when it is
// longer than one pass. Placeholder tokens of embeds take its rows.
func (w *gpuWorkspace) batch(m *Weights, seqs [][]int, dst [][]float32, pre *gpuPrefix, past int, shared bool, embeds Embeds) error {
	c := &m.cfg
	own := 4 * gpuPositions * c.kvDim
	w.curK, w.curV, w.curStride = w.kc, w.vc, own
	w.preK, w.preV, w.preStride = w.kc, w.vc, own
	w.attn.base, w.attn.prefixLen = 0, 0
	if pre != nil {
		stride := 4 * pre.capacity * c.kvDim
		if shared {
			w.preK, w.preV, w.preStride = pre.kc, pre.vc, stride
			w.attn.prefixLen = uint32(past)
		} else {
			w.curK, w.curV, w.curStride = pre.kc, pre.vc, stride
			w.attn.base = uint32(past)
		}
	}
	w.past, w.shared, w.embeds, w.spliced = past, shared, embeds, 0
	defer func() { w.embeds, w.oneSeq[0] = Embeds{}, nil }()
	if pre != nil && !shared && len(seqs) == 1 && len(seqs[0]) > w.rows {
		ids := seqs[0]
		for off := 0; off < len(ids); off += w.rows {
			w.oneSeq[0] = ids[off:min(off+w.rows, len(ids))]
			if err := w.pass(m, w.oneSeq[:], dst, len(w.oneSeq[0])); err != nil {
				return err
			}
			w.past += len(w.oneSeq[0])
			w.attn.base = uint32(w.past)
		}
		return nil
	}
	for start := 0; start < len(seqs); {
		end, rows := start, 0
		for end < len(seqs) && rows+len(seqs[end]) <= w.rows {
			rows += len(seqs[end])
			end++
		}
		if end == start {
			return fmt.Errorf("qwen3: %d tokens exceeds the GPU context %d", len(seqs[start]), w.rows)
		}
		if err := w.pass(m, seqs[start:end], dst[start:end], rows); err != nil {
			return err
		}
		start = end
	}
	return nil
}

// logits writes the head's logits for one post-final-norm state.
func (w *gpuWorkspace) logitsInto(m *Weights, hidden, dst []float32) error {
	g, c := w.g, &m.cfg
	x := floats(w.h.Bytes())[:c.hidden]
	copy(x, hidden)
	g.hidden.apply(x)
	e := &w.enc
	g.dev.Begin(e, false)
	w.gemv(g.gemvHead, g.lm, 0, g.lmScale, w.h, 0, w.logits, 0, w.attnParts, w.attnParts, 0, &w.headArgs)
	if err := e.Wait(); err != nil {
		return err
	}
	copy(dst, floats(w.logits.Bytes())[:c.vocab])
	return nil
}

func (w *gpuWorkspace) pass(m *Weights, seqs [][]int, dst [][]float32, rows int) error {
	g, c := w.g, &m.cfg
	hs := floats(w.h.Bytes())
	embedParts := floats(w.embedParts.Bytes())
	info := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(w.info.Bytes()))), 2*gpuPositions)
	r := 0
	for _, ids := range seqs {
		start := r
		for pos, id := range ids {
			row := hs[r*c.hidden : (r+1)*c.hidden]
			if len(w.embeds.Rows) != 0 && id == w.embeds.Token {
				copy(row, w.embeds.Rows[w.spliced*c.hidden:])
				w.spliced++
			} else {
				m.embedRow(id, row)
			}
			g.hidden.apply(row)
			embedParts[r] = sumSquares(row)
			info[2*r], info[2*r+1] = uint32(w.past+pos), uint32(start)
			r++
		}
	}
	e := &w.enc
	g.dev.Begin(e, false)
	if (gpuTokenByToken || rows == 1) && !w.shared {
		if len(seqs) != 1 {
			panic("qwen3: token-by-token GPU path takes one sequence")
		}
		w.encodeTokens(rows)
	} else {
		w.encodeBatch(rows)
	}
	if err := e.Wait(); err != nil {
		return err
	}
	r = 0
	for s, ids := range seqs {
		r += len(ids)
		out := dst[s]
		copy(out, hs[(r-1)*c.hidden:r*c.hidden])
		g.hidden.unapply(out)
		rmsNorm32(out, out, m.finalNorm, c.eps)
		for _, v := range out {
			if !finite32(v) {
				return errors.New("qwen3: non-finite hidden state")
			}
		}
	}
	return nil
}

// mm encodes one batched projection over rows tokens.
func (w *gpuWorkspace) mmDispatch(kind int, buf *metal.Buffer, wOff, sOff int, x, y, in, out *metal.Buffer,
	k, n, rows, parts int) {
	e := &w.enc
	// Up to 16 tokens use the 16-token tile, which wastes no rows.
	tile, p := 32, w.g.mm[0][kind]
	if rows <= 16 {
		tile, p = 16, w.g.mm[1][kind]
	}
	groups := n / mmColumns * ((rows + tile - 1) / tile)
	// Small grids leave GPU cores idle; split K so at least mmMinGroups
	// threadgroups stream the weights, and add the splits in mm_finish.
	splits := 1
	for splits < 8 && groups*splits < mmMinGroups && k%(2*splits*32) == 0 {
		splits *= 2
	}
	padM := (rows + tile - 1) / tile * tile
	w.mm = mmArgs{k: uint32(k), n: uint32(n), m: uint32(rows), eps: float32(w.g.cfg.eps), parts: uint32(parts),
		partsOut: uint32(n / mmColumns), splitK: uint32(k / splits), splits: uint32(splits), padM: uint32(padM)}
	e.SetPipeline(p)
	e.SetBuffer(buf, wOff, 0)
	e.SetBuffer(buf, sOff, 1)
	e.SetBuffer(x, 0, 2)
	e.SetBuffer(y, 0, 3)
	e.SetBuffer(in, 0, 4)
	e.SetBytes(unsafe.Pointer(&w.mm), int(unsafe.Sizeof(w.mm)), 5)
	e.SetBuffer(out, 0, 6)
	e.SetBuffer(w.scratch, 0, 7)
	e.Dispatch(metal.Size{X: n / mmColumns, Y: (rows + tile - 1) / tile, Z: splits}, metal.Size{X: 8 * tile, Y: 1, Z: 1})
	if splits > 1 {
		e.SetPipeline(w.g.finish[[4]int{0, 1, 2, 1}[kind]])
		e.Dispatch(metal.Size{X: n / mmColumns, Y: rows, Z: 1}, metal.Size{X: mmColumns, Y: 1, Z: 1})
	}
}

// encodeBatch encodes a forward pass over rows tokens at positions 0..rows-1
// with batched projections, reading each weight once per 32 tokens.
func (w *gpuWorkspace) encodeBatch(rows int) {
	g, c, e := w.g, w.g.cfg, &w.enc
	qdim := c.heads * c.headDim
	parts := c.hidden / mmColumns
	w.attn.pos = 0
	w.perRow = uint32(c.intermediate / g.inter.block)
	for i := range g.layers {
		gl := &g.layers[i]
		if i == 0 {
			w.mmDispatch(0, gl.buf, gl.qkv, gl.qkvScale, w.h, w.qkv, w.embedParts, w.attnParts, c.hidden, qdim+2*c.kvDim, rows, 1)
		} else {
			w.mmDispatch(0, gl.buf, gl.qkv, gl.qkvScale, w.h, w.qkv, w.attnParts, w.attnParts, c.hidden, qdim+2*c.kvDim, rows, parts)
		}
		e.SetPipeline(g.qkRope)
		e.SetBuffer(w.qkv, 0, 0)
		e.SetBuffer(w.curK, i*w.curStride, 1)
		e.SetBuffer(w.curV, i*w.curStride, 2)
		e.SetBuffer(g.norms, 4*2*i*c.headDim, 3)
		e.SetBuffer(g.norms, 4*(2*i+1)*c.headDim, 4)
		e.SetBuffer(g.rope, 0, 5)
		e.SetBuffer(w.info, 0, 6)
		e.SetBytes(unsafe.Pointer(&w.attn), int(unsafe.Sizeof(w.attn)), 7)
		e.Dispatch(metal.Size{X: c.heads + 2*c.kvHeads, Y: rows, Z: 1}, metal.Size{X: 32, Y: 1, Z: 1})

		e.SetBuffer(w.info, 0, 3)
		e.SetBuffer(w.preK, i*w.preStride, 4)
		e.SetBuffer(w.preV, i*w.preStride, 5)
		e.SetBuffer(w.ctx, 0, 6)
		if gpuScalarAttention {
			e.SetPipeline(g.attendM)
			e.Dispatch(metal.Size{X: c.kvHeads, Y: rows, Z: 1}, metal.Size{X: 32 * c.heads / c.kvHeads, Y: 1, Z: 1})
		} else {
			w.batchRows = uint32(rows)
			e.SetPipeline(g.attendFlash)
			e.SetBytes(unsafe.Pointer(&w.batchRows), 4, 8)
			e.Dispatch(metal.Size{X: c.kvHeads, Y: (rows + 7) / 8, Z: 1}, metal.Size{X: 32 * c.heads / c.kvHeads, Y: 1, Z: 1})
		}

		w.mmDispatch(1, gl.buf, gl.o, gl.oScale, w.ctx, w.h, w.mlpParts, w.mlpParts, qdim, c.hidden, rows, 0)
		w.mmDispatch(2, gl.buf, gl.gu, gl.guScale, w.h, w.act, w.mlpParts, w.mlpParts, c.hidden, 2*c.intermediate, rows, parts)

		if !g.noInter {
			e.SetPipeline(g.rotate)
			e.SetBuffer(w.act, 0, 0)
			e.SetBuffer(g.signs, 0, 1)
			e.SetBytes(unsafe.Pointer(&w.perRow), 4, 2)
			e.Dispatch(metal.Size{X: rows * int(w.perRow), Y: 1, Z: 1}, metal.Size{X: g.inter.block / 4, Y: 1, Z: 1})
		}

		w.mmDispatch(3, gl.buf, gl.d, gl.dScale, w.act, w.h, w.attnParts, w.attnParts, c.intermediate, c.hidden, rows, 0)
	}
}

// encodeTokens encodes the forward pass one token at a time with GEMV
// kernels, which stream weights fastest for a single token.
func (w *gpuWorkspace) encodeTokens(n int) {
	g, c, e := w.g, w.g.cfg, &w.enc
	w.perRow = uint32(c.intermediate / g.inter.block)
	for t := range n {
		hOff := 4 * t * c.hidden
		w.attn.pos = uint32(w.past + t)
		for i := range g.layers {
			gl := &g.layers[i]
			if i == 0 {
				w.gemv(g.qkv, gl.buf, gl.qkv, gl.qkvScale, w.h, hOff, w.qkv, 0, w.embedParts, w.attnParts, 4*t, &w.qkv0Args)
			} else {
				w.gemv(g.qkv, gl.buf, gl.qkv, gl.qkvScale, w.h, hOff, w.qkv, 0, w.attnParts, w.attnParts, 0, &w.qkvArgs)
			}

			e.SetPipeline(g.attend)
			e.SetBuffer(w.qkv, 0, 0)
			e.SetBuffer(w.curK, i*w.curStride, 1)
			e.SetBuffer(w.curV, i*w.curStride, 2)
			e.SetBuffer(g.norms, 4*2*i*c.headDim, 3)
			e.SetBuffer(g.norms, 4*(2*i+1)*c.headDim, 4)
			e.SetBuffer(g.rope, 0, 5)
			e.SetBuffer(w.ctx, 0, 6)
			e.SetBytes(unsafe.Pointer(&w.attn), int(unsafe.Sizeof(w.attn)), 7)
			e.Dispatch(metal.Size{X: c.kvHeads, Y: 1, Z: 1}, metal.Size{X: 32 * attendSplits * c.heads / c.kvHeads, Y: 1, Z: 1})

			w.gemv(g.o, gl.buf, gl.o, gl.oScale, w.ctx, 0, w.h, hOff, w.mlpParts, w.mlpParts, 0, &w.oArgs)
			w.gemv(g.gateup, gl.buf, gl.gu, gl.guScale, w.h, hOff, w.act, 0, w.mlpParts, w.mlpParts, 0, &w.guArgs)

			if !g.noInter {
				e.SetPipeline(g.rotate)
				e.SetBuffer(w.act, 0, 0)
				e.SetBuffer(g.signs, 0, 1)
				e.SetBytes(unsafe.Pointer(&w.perRow), 4, 2)
				e.Dispatch(metal.Size{X: int(w.perRow), Y: 1, Z: 1}, metal.Size{X: g.inter.block / 4, Y: 1, Z: 1})
			}

			w.gemv(g.down, gl.buf, gl.d, gl.dScale, w.act, 0, w.h, hOff, w.attnParts, w.attnParts, 0, &w.dArgs)
		}
	}
}

func (w *gpuWorkspace) release() {
	for _, b := range []*metal.Buffer{w.h, w.qkv, w.ctx, w.act, w.kc, w.vc, w.embedParts, w.info, w.scratch, w.attnParts, w.mlpParts, w.logits} {
		b.Release()
	}
}

// Release frees the model's GPU buffers; the Weights are unusable after.
func (m *Weights) Release() {
	g := m.gpu
	if g == nil {
		return
	}
	for i := range g.layers {
		g.layers[i].buf.Release()
	}
	if g.lm != nil {
		g.lm.Release()
	}
	g.norms.Release()
	g.signs.Release()
	g.rope.Release()
	g.dev.Close()
	m.gpu = nil
}

func (g *gpuModel) hasHead() bool { return g.lm != nil }

// maxPositions is the most positions a GPU prefix can hold.
func (g *gpuModel) maxPositions() int { return g.positions }

// GPUAvailable reports whether a Metal GPU is present.
func GPUAvailable() bool {
	gpuProbe.Do(func() {
		if d, err := metal.Open(); err == nil {
			d.Close()
			gpuPresent = true
		}
	})
	return gpuPresent
}
