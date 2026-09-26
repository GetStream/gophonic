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
	"sync/atomic"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/internal/wcache"
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
	// gpuCacheVersion names the layout of prepared weights in the cache;
	// change it with anything that changes their bytes.
	gpuCacheVersion = "qwen3lm-gpu-1"
	gpuHeadRows     = 16 // GEMV_ROWS_HEAD (8 simdgroups × 2 rows)
)

// gpuLayer holds one layer's projections in one shared buffer: int8 rows
// followed by FP32 row scales, at byte offsets.
type gpuLayer struct {
	buf                                              *metal.Buffer
	qkv, qkvScale, o, oScale, gu, guScale, d, dScale int
	// router is the FP32 router of a mixture of experts; gu and d then hold
	// every expert's gate/up and down rows, expert after expert.
	router int
	// In the Qwen3.5 family, linear marks a DeltaNet layer; index is the
	// layer's place among its kind, which picks its key/value cache or its
	// state and parameters.
	linear bool
	index  int
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
	probe                        *metal.Pipeline // one head's attention, for a Probe
	decodeHead, decodeSample     *metal.Pipeline // a Decoder's steps
	decodeGather                 *metal.Pipeline
	decodeVecOnce                sync.Once
	decodeVecErr                 error
	decodeVec                    [4][5]*metal.Pipeline // widths 1, 2, 4, 8; qkv, o, gateup, down, head
	greedyArgmax                 *metal.Pipeline
	mm                           [2][4]*metal.Pipeline // [32-token, 16-token tiles][qkv, o, gateup, down]
	finish                       [3]*metal.Pipeline    // split-K epilogues: store, add, SwiGLU
	mmHead                       [2]*metal.Pipeline    // the head over several rows: 32-token, 16-token tiles
	finishHead                   *metal.Pipeline
	moeRouter, moeRoute          *metal.Pipeline // a mixture of experts' router and top-k
	moeGateUp, moeDown           *metal.Pipeline // one row's experts
	// Batches grouped by expert: the tiles, the 32- and 16-pair GEMMs, and
	// the sum of each row's experts.
	moeTiles, moeCombine   *metal.Pipeline
	moeGateUpMM, moeDownMM [2]*metal.Pipeline
	hy                     hybridPipelines // the Qwen3.5 family's kernels
	dnParams               *metal.Buffer   // DeltaNet layers' parameters
	layers                 []gpuLayer
	norms, signs, rope     *metal.Buffer
	hidden, inter, head    *rotation
	cfg                    *modelConfig
	noInter                bool // down's inputs are not rotated
	bits                   int  // weight scheme, as in gpu.metal: 8 (Q8), 9 (Q8B), or 4 (Q4)
	positions              int  // RoPE table rows
	// lm is the language-model head W·Rᵀ as Q8B int8 blocks with FP16
	// scales, padded with zero rows to lmRows; nil unless loaded.
	lm        *metal.Buffer
	lmScale   int // byte offset of the scales in lm
	lmRows    int
	cache     *wcache.File // holds the layer buffers' memory
	headCache *wcache.File // holds the head's, once it is loaded
}

// gpuRows is the number of projection rows per GEMV threadgroup: 8
// simdgroups times rowsPerSimdgroup in gpu.metal.
func gpuRows(bits int) int {
	if bits == 8 || bits == 9 {
		return 16
	}
	return 32
}

func alignUp(n int) int { return (n + gpuAlign - 1) &^ (gpuAlign - 1) }

// gpuGeometry checks that the GPU kernels, which gpu.metal specializes by
// the model's widths, fit the model: 128-wide heads, at most eight query
// heads per key/value head, and projection widths that tile evenly.
func gpuGeometry(c *modelConfig) error {
	if c.hybrid {
		return hybridGeometry(c)
	}
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
	return gpuSourceMode(c, false)
}

func gpuDecodeBatchSourceFor(c *modelConfig) string {
	return gpuSourceMode(c, true)
}

func gpuSourceMode(c *modelConfig, decodeBatch bool) string {
	header := fmt.Sprintf("#define QD %d\n#define KVD %d\n#define NH %d\n#define NKV %d\n#define ROT %d\n",
		c.heads*c.headDim, c.kvDim, c.heads, c.kvHeads, newRotation(c.intermediate).block)
	if decodeBatch {
		header += "#define QWEN3_DECODE_BATCH\n"
	}
	return header + gpuSource
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
func (m *Weights) loadGPU(st *safetensors.Checkpoint, bits int) error {
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
	m.gpu = g // Release frees what is loaded when loading fails
	suffix := map[int]string{8: "", 9: "_q8", 4: "_q4"}[bits]
	src := gpuSourceFor(c)
	if c.hybrid {
		src += hybridDefines(c) + gpuHybridSource
	}
	lib, err := dev.Compile(src)
	if err != nil {
		return err
	}
	defer lib.Release()
	for _, p := range []struct {
		dst  **metal.Pipeline
		name string
	}{{&g.qkv, "gemv_qkv" + suffix}, {&g.o, "gemv_o" + suffix}, {&g.gateup, "gemv_gateup" + suffix}, {&g.down, "gemv_down" + suffix}, {&g.attend, "attend1"}, {&g.probe, "probe1"}, {&g.rotate, "rotate"},
		{&g.decodeHead, "decode_head"}, {&g.decodeSample, "decode_sample"}, {&g.decodeGather, "decode_gather"},
		{&g.gemvHead, "gemv_head"},
		{&g.qkRope, "qkRope"}, {&g.attendM, "attendM"}, {&g.attendFlash, "attendFlash"},
		{&g.mm[0][0], "mm_qkv" + suffix + "_w"}, {&g.mm[0][1], "mm_o" + suffix + "_w"}, {&g.mm[0][2], "mm_gateup" + suffix + "_w"}, {&g.mm[0][3], "mm_down" + suffix + "_w"},
		{&g.finish[0], "mm_finish_store" + suffix}, {&g.finish[1], "mm_finish_add" + suffix}, {&g.finish[2], "mm_finish_swiglu" + suffix},
		{&g.mmHead[0], "mm_head_w"}, {&g.mmHead[1], "mm_head_16"}, {&g.finishHead, "mm_finish_store_q8"},
		{&g.mm[1][0], "mm_qkv" + suffix + "_16"}, {&g.mm[1][1], "mm_o" + suffix + "_16"}, {&g.mm[1][2], "mm_gateup" + suffix + "_16"}, {&g.mm[1][3], "mm_down" + suffix + "_16"},
		{&g.moeRouter, "moe_router"}, {&g.moeRoute, "moe_route"}, {&g.moeGateUp, "moe_gateup_q8"}, {&g.moeDown, "moe_down_q8"},
		{&g.moeTiles, "moe_tiles"}, {&g.moeCombine, "moe_combine"},
		{&g.moeGateUpMM[0], "moe_gateup_mm_w"}, {&g.moeGateUpMM[1], "moe_gateup_mm_16"},
		{&g.moeDownMM[0], "moe_down_mm_w"}, {&g.moeDownMM[1], "moe_down_mm_16"}} {
		if *p.dst, err = dev.Pipeline(lib, p.name); err != nil {
			return err
		}
	}
	if c.hybrid {
		if err := g.hybridPipelines(lib); err != nil {
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
	half := len(c.invFreq) // rotary dimensions / 2
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

	// Each layer's projections share one buffer: int8 or 4-bit rows, then
	// their scales. Layers of a kind have the same offsets.
	shapes, sizes := make([]gpuLayer, c.layers), make([]int, c.layers)
	counts := [2]int{} // attention and DeltaNet layers so far
	for i := range shapes {
		layerBytes := 0
		place := func(rows, k int) (int, int) {
			w := layerBytes
			if bits == 4 {
				layerBytes = alignUp(layerBytes + rows*k/2)
			} else {
				layerBytes = alignUp(layerBytes + rows*k)
			}
			s := layerBytes
			if bits == 8 {
				layerBytes = alignUp(layerBytes + 4*rows)
			} else {
				layerBytes = alignUp(layerBytes + 2*rows*(k/q4Group))
			}
			return w, s
		}
		shape := &shapes[i]
		shape.linear = c.hybrid && c.linear[i]
		kind := 0
		if shape.linear {
			kind = 1
		}
		shape.index = counts[kind]
		counts[kind]++
		switch {
		case shape.linear:
			shape.qkv, shape.qkvScale = place(c.dnIn(), h)
			shape.o, shape.oScale = place(h, c.dnValueHeads*c.dnValueDim)
		case c.hybrid: // q and its gate, head by head, then k and v
			shape.qkv, shape.qkvScale = place(2*qdim+2*kv, h)
			shape.o, shape.oScale = place(h, qdim)
		default:
			shape.qkv, shape.qkvScale = place(qdim+2*kv, h)
			shape.o, shape.oScale = place(h, qdim)
		}
		if e := c.allExperts(); e > 0 {
			shape.router = layerBytes
			layerBytes = alignUp(layerBytes + 4*routerRows(c)*h)
			shape.gu, shape.guScale = place(e*2*inter, h)
			shape.d, shape.dScale = place(e*h, inter)
		} else {
			shape.gu, shape.guScale = place(2*inter, h)
			shape.d, shape.dScale = place(h, inter)
		}
		sizes[i] = layerBytes
	}
	// The buffers are regions of a cache entry: quantized on the first load,
	// mapped as they are afterwards. The head has an entry of its own.
	cut := func(l *wcache.Layout) (layers [][]byte) {
		for _, n := range sizes {
			layers = append(layers, l.Take(n))
		}
		return layers
	}
	size := wcache.NewLayout(nil)
	cut(size)
	key, err := wcache.Key(st.Dir(), m.cacheKind(""), gpuCacheVersion, m.prefix, "")
	if err != nil {
		return err
	}
	if g.cache, err = wcache.Open(key, size.Size()); err != nil {
		return err
	}
	if g.cache.Fresh() {
		if err := m.prepareGPU(st, g, shapes, cut(wcache.NewLayout(g.cache.Payload())), nil, downIn, "", g.cache.Done); err != nil {
			return err
		}
		g.cache.Commit()
	}
	// The GPU reads the committed, read-only entry in place.
	layers := cut(wcache.NewLayout(g.cache.Payload()))
	for i := range g.layers {
		g.layers[i] = shapes[i]
		if g.layers[i].buf, err = dev.Wrap(layers[i]); err != nil {
			return err
		}
	}
	if c.hybrid {
		return g.uploadDeltaNet(m)
	}
	return nil
}

// loadHead quantizes the head named name, W·Rᵀ as int8 blocks (Q8B), into
// a cache entry of its own beside the layers': LogitsInto rotates the
// normalized state.
func (g *gpuModel) loadHead(m *Weights, st *safetensors.Checkpoint, name string) error {
	c := &m.cfg
	h := c.hidden
	rows := (c.vocab + gpuRows(9) - 1) / gpuRows(9) * gpuRows(9)
	scale := alignUp(rows * h)
	n := scale + 2*rows*(h/q4Group)
	key, err := wcache.Key(st.Dir(), m.cacheKind("-lm"), gpuCacheVersion, m.prefix, name)
	if err != nil {
		return err
	}
	size := wcache.NewLayout(nil)
	size.Take(n)
	cache, err := wcache.Open(key, size.Size())
	if err != nil {
		return err
	}
	g.lmRows, g.lmScale = rows, scale
	if cache.Fresh() {
		if err := m.prepareGPU(st, g, nil, nil, wcache.NewLayout(cache.Payload()).Take(n), nil, name, cache.Done); err != nil {
			cache.Close()
			g.lmRows, g.lmScale = 0, 0
			return err
		}
		cache.Commit()
	}
	if g.lm, err = g.dev.Wrap(wcache.NewLayout(cache.Payload()).Take(n)); err != nil {
		cache.Close()
		g.lmRows, g.lmScale = 0, 0
		return err
	}
	g.headCache = cache
	return nil
}

// gpuJob quantizes one matrix, or a chunk of one, into a layer's region.
type gpuJob struct {
	name     string
	n, k     int
	norm     []float32 // folded input RMSNorm weight
	in, out  *rotation // input- and output-side rotations; out spans a chunk
	buf      []byte
	base, sc int // byte offsets of row 0 and scale 0
	step     int // destination row stride in rows
	scaleMul float32
	row0     int    // destination row of source row 0
	rows     [2]int // for a chunk of a large matrix: its source rows
	f32      bool   // stored as FP32 rows at base, unquantized
	dims     []int  // the tensor's shape when not [n][k], as stacked experts'
}

// prepareGPU quantizes every projection into its layer's region (layers
// may be none), and the head, when named, into lm.
func (m *Weights) prepareGPU(st *safetensors.Checkpoint, g *gpuModel, shapes []gpuLayer, layers [][]byte, lm []byte, downIn *rotation, headName string, filled func([]byte)) error {
	c := &m.cfg
	bits := g.bits
	h, kv, inter, qdim := c.hidden, c.kvDim, c.intermediate, c.heads*c.headDim

	type job = gpuJob
	var jobs []job
	for i := range layers {
		l, gl, buf := &m.layers[i], &shapes[i], layers[i]
		p := fmt.Sprintf("%slayers.%d.", m.prefix, i)
		if c.hybrid {
			jobs = m.hybridJobs(jobs, g, i, gl, buf, downIn)
			continue
		}
		jobs = append(jobs,
			job{p + "self_attn.q_proj.weight", qdim, h, l.attnNorm, g.hidden, nil, buf, gl.qkv, gl.qkvScale, 1, 1, 0, [2]int{}, false, nil},
			job{p + "self_attn.k_proj.weight", kv, h, l.attnNorm, g.hidden, nil, buf, gl.qkv, gl.qkvScale, 1, 1, qdim, [2]int{}, false, nil},
			job{p + "self_attn.v_proj.weight", kv, h, l.attnNorm, g.hidden, g.head, buf, gl.qkv, gl.qkvScale, 1, 1, qdim + kv, [2]int{}, false, nil},
			job{p + "self_attn.o_proj.weight", h, qdim, nil, g.head, g.hidden, buf, gl.o, gl.oScale, 1, 1, 0, [2]int{}, false, nil},
		)
		if c.experts == 0 {
			jobs = append(jobs,
				job{p + "mlp.gate_proj.weight", inter, h, l.mlpNorm, g.hidden, nil, buf, gl.gu, gl.guScale, 2, 1, 0, [2]int{}, false, nil},
				job{p + "mlp.up_proj.weight", inter, h, l.mlpNorm, g.hidden, nil, buf, gl.gu, gl.guScale, 2, 1, 1, [2]int{}, false, nil},
				job{p + "mlp.down_proj.weight", h, inter, nil, downIn, g.hidden, buf, gl.d, gl.dScale, 1, 1, 0, [2]int{}, false, nil},
			)
			continue
		}
		// The router reads the normalized residual as gate/up does; each
		// expert's gate and up rows alternate, expert after expert.
		jobs = append(jobs, job{p + "mlp.gate.weight", c.experts, h, l.mlpNorm, g.hidden, nil, buf, gl.router, 0, 1, 1, 0, [2]int{}, true, nil})
		for e := range c.experts {
			q := fmt.Sprintf("%smlp.experts.%d.", p, e)
			jobs = append(jobs,
				job{q + "gate_proj.weight", inter, h, l.mlpNorm, g.hidden, nil, buf, gl.gu, gl.guScale, 2, 1, e * 2 * inter, [2]int{}, false, nil},
				job{q + "up_proj.weight", inter, h, l.mlpNorm, g.hidden, nil, buf, gl.gu, gl.guScale, 2, 1, e*2*inter + 1, [2]int{}, false, nil},
				job{q + "down_proj.weight", h, inter, nil, downIn, g.hidden, buf, gl.d, gl.dScale, 1, 1, e * h, [2]int{}, false, nil},
			)
		}
	}
	// The head, the largest matrix, is read in chunks of at most one
	// projection.
	chunk := max(inter, 4096)
	for r := 0; headName != "" && r < c.vocab; r += chunk {
		jobs = append(jobs, job{headName, c.vocab, h, nil, g.hidden, nil, lm, 0, g.lmScale, 1, 1, 0, [2]int{r, min(r+chunk, c.vocab)}, false, nil})
	}
	// GPTQ-rounded weights made by QuantizeGPTQ replace round-to-nearest.
	var pre *gptqFile
	if format := map[int]string{8: WeightsGPU, 4: WeightsGPUQ4}[bits]; format != "" {
		if pre = openGPTQ(st.Dir(), format, bits, c); pre != nil {
			defer pre.close()
		}
	}
	// A region is filled when its last job finishes.
	region := map[*byte]int{}
	left := make([]atomic.Int32, len(layers)+1)
	for _, j := range jobs {
		r, ok := region[&j.buf[0]]
		if !ok {
			r = len(region)
			region[&j.buf[0]] = r
		}
		left[r].Add(1)
	}
	finish := func(j job) {
		if left[region[&j.buf[0]]].Add(-1) == 0 {
			filled(j.buf)
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
					done, err := pre.place(strings.TrimPrefix(j.name, m.prefix), j.n, j.k, j.buf, j.base, j.sc, j.row0, j.step)
					if err != nil {
						mu.Lock()
						first = errors.Join(first, err)
						mu.Unlock()
						return
					}
					if done {
						finish(j)
						continue
					}
				}
				r0, n := 0, j.n
				if j.rows[1] > 0 {
					r0, n = j.rows[0], j.rows[1]-j.rows[0]
				}
				dims := j.dims
				if dims == nil {
					dims = []int{j.n, j.k}
				}
				t, err := st.Lookup(j.name, dims...)
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
					j.out.applyRows(mat, n, j.k)
				}
				buf := j.buf
				if j.f32 {
					copy(floats(buf[j.base:])[(j.row0+r0*j.step)*j.k:], mat[:n*j.k])
					finish(j)
					continue
				}
				for r := range n {
					row := mat[r*j.k : (r+1)*j.k]
					dst := j.row0 + (r0+r)*j.step
					scheme := bits
					if j.name == headName {
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
				finish(j)
			}
		}()
	}
	wg.Wait()
	return first
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
	logitRows                               int // rows the logits buffer holds
	headArgs                                gemvArgs
	tail                                    []float32 // a single sequence's last states, when set
	// A Decoder run's tokens and logits, grown to the largest run.
	decodeTokens, decodeLogits *metal.Buffer
	decodeArgs                 decodeArgs
	probe                      *Probe // the head a one-token pass reads, when set
	probeOut                   *metal.Buffer
	probeArgs                  probeArgs
	oneSeq                     [1][]int
	// A mixture of experts: the router's logits, each row's experts and
	// weights, and the arguments of its kernels. Batches group their (row,
	// slot) pairs by expert: each expert's count, each pair's rank among its
	// expert's, the pairs expert after expert, the GEMM tiles over them, and
	// each pair's weighted output.
	route, ids, wts               *metal.Buffer
	counts, rank, list, tiles     *metal.Buffer
	moeOut                        *metal.Buffer
	moeRouter, moeGateUp, moeDown moeArgs
	moeTiles                      moeTileArgs
	hy                            hybridWork
}

// moeTileArgs is MoeTileArgs in gpu.metal.
type moeTileArgs struct {
	pairs, tile, tiles uint32
}

// moeGroupRows is the batch size from which experts run grouped: below it,
// each row streams its own experts' weights.
var moeGroupRows = 16

// moeArgs is MoeArgs in gpu.metal.
type moeArgs struct {
	k, n     uint32
	eps      float32
	parts    uint32
	experts  uint32
	topK     uint32
	partsOut uint32
	stride   uint32
	shared   uint32
}

// gpuDecodeBatchWorkspace owns scratch for one synchronous set of independent
// single-token decodes. K/V storage remains in each lane's gpuPrefix.
type gpuDecodeBatchWorkspace struct {
	g                               *gpuModel
	capacity, vectorWidth           int
	laneGroups                      int
	h, qkv, ctx, act, logits        *metal.Buffer
	tokenIDs                        *metal.Buffer
	embedParts, attnParts, mlpParts *metal.Buffer
	finalHidden                     []float32
	enc                             metal.Encoder
	qkv0Args, qkvArgs               gemvArgs
	oArgs, guArgs, dArgs, headArgs  gemvArgs
	argmaxArgs                      greedyArgmaxArgs
	attn                            attnArgs
	perRow                          uint32
	concurrent                      bool
	argmaxGPU                       bool
}

// gpuTokenByToken forces the single-token kernels and gpuScalarAttention
// the per-key attention loop; tests compare the paths.
var gpuTokenByToken, gpuScalarAttention bool

// attendSplits is the simdgroups that share one query head's keys in the
// single-token attention (AS in gpu.metal).
const attendSplits = 4

// mmColumns is the GEMM tile width in weight rows (MM_BN in gpu.metal), and
// mmThreads the threads of both GEMM tiles.
const mmColumns, mmThreads = 64, 128

type mmArgs struct {
	k, n, m         uint32
	eps             float32
	parts, partsOut uint32
	splitK, splits  uint32
	padM            uint32
}

// mmMinGroups is the threadgroup count below which projections split K.
const mmMinGroups = 256

// mmScratchFloats bounds split-K scratch. The final power-of-two split has
// fewer than 2*mmMinGroups tiles; each tile covers at most 32×mmColumns
// outputs. A one-split GEMM never writes scratch.
const mmScratchFloats = 2 * mmMinGroups * 32 * mmColumns

type gemvArgs struct {
	k, n  uint32
	eps   float32
	parts uint32
}

type greedyArgmaxArgs struct {
	vocab, stride uint32
}

// probeArgs is ProbeArgs in gpu.metal.
type probeArgs struct {
	head, from, n uint32
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
	state    *gpuState // a hybrid model's recurrent state after the prefix
	// Snapshots of state where extensions began (hybrid_state.go), states
	// no longer needed, and a clock that orders their use.
	snaps, spare []*gpuState
	clock        uint64
}

// copyStates makes p's state and snapshots src's, up to token n.
func (p *gpuPrefix) copyStates(src *gpuPrefix, n int) {
	p.state.copyFrom(src.state)
	p.spare = append(p.spare, p.snaps...)
	p.snaps = p.snaps[:0]
	for _, s := range src.snaps {
		if s.pos > n {
			continue
		}
		var d *gpuState
		if len(p.spare) > 0 {
			d, p.spare = p.spare[len(p.spare)-1], p.spare[:len(p.spare)-1]
		} else if d, _ = newStateLike(src.state); d == nil {
			return // the state rewinds further back instead
		}
		d.copyFrom(s)
		p.snaps = append(p.snaps, d)
	}
}

func (g *gpuModel) newPrefix(capacity int) (*gpuPrefix, error) {
	n := 4 * g.cfg.attnLayers() * capacity * g.cfg.kvDim
	kc, err := g.dev.Buffer(n)
	if err != nil {
		return nil, err
	}
	vc, err := g.dev.Buffer(n)
	if err != nil {
		kc.Release()
		return nil, err
	}
	p := &gpuPrefix{kc: kc, vc: vc, capacity: capacity}
	if g.cfg.hybrid {
		if p.state, err = g.newState(); err != nil {
			p.release()
			return nil, err
		}
	}
	return p, nil
}

// ensureKV allocates the workspace's temporary cache only for passes that
// own their current K/V rows. Prefix continuation writes directly into the
// lane's PrefixKV and does not need these full-context buffers.
func (w *gpuWorkspace) ensureKV() error {
	if w.kc != nil && w.vc != nil {
		return nil
	}
	n := 4 * w.g.cfg.attnLayers() * gpuPositions * w.g.cfg.kvDim
	kc, err := w.g.dev.Buffer(n)
	if err != nil {
		return err
	}
	vc, err := w.g.dev.Buffer(n)
	if err != nil {
		kc.Release()
		return err
	}
	w.kc, w.vc = kc, vc
	return nil
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
	if p.state != nil {
		p.state.release()
	}
	for _, s := range append(p.snaps, p.spare...) {
		s.release()
	}
	p.snaps, p.spare = nil, nil
}

func (g *gpuModel) newWorkspace() (*gpuWorkspace, error) {
	c := g.cfg
	qdim := c.heads * c.headDim
	rows := gpuRows(g.bits)
	w := &gpuWorkspace{g: g, rows: gpuPositions}
	ok := false
	defer func() {
		if !ok {
			w.release()
		}
	}()
	var err error
	// Each row's experts write their own activations; their down kernel
	// publishes a partial sum per 16 rows of the residual.
	// A hybrid layer's input projection is gated attention's or DeltaNet's.
	qkvWidth := qdim + 2*c.kvDim
	if c.hybrid {
		qkvWidth = max(2*qdim+2*c.kvDim, c.dnIn())
	}
	// Grouped, each (row, slot) pair keeps its expert's output.
	perRow, rowParts, experts, route, pairs, pairOut := c.intermediate, c.hidden/mmColumns, 1, 1, gpuPositions, 1
	if c.experts > 0 {
		perRow, rowParts, experts, route = c.slots()*c.intermediate, c.hidden/rows, c.allExperts(), routerRows(c)
		pairs *= c.slots()
		pairOut = pairs * c.hidden
	}
	for _, b := range []struct {
		dst **metal.Buffer
		n   int
	}{
		{&w.h, 4 * gpuPositions * c.hidden},
		{&w.qkv, 4 * gpuPositions * qkvWidth},
		{&w.ctx, 4 * gpuPositions * max(qdim, c.dnValueHeads*c.dnValueDim)},
		{&w.act, 4 * gpuPositions * perRow},
		{&w.embedParts, 4 * gpuPositions},
		{&w.info, 8 * gpuPositions},
		{&w.scratch, 4 * mmScratchFloats},
		{&w.attnParts, 4 * max(c.hidden/rows, gpuPositions*rowParts)},
		{&w.mlpParts, 4 * max(c.hidden/rows, gpuPositions*c.hidden/mmColumns)},
		{&w.logits, 4 * max(g.lmRows, 1)},
		{&w.route, 4 * gpuPositions * route},
		{&w.ids, 4 * pairs},
		{&w.wts, 4 * pairs},
		{&w.counts, 4 * experts},
		{&w.rank, 4 * pairs},
		{&w.list, 4 * pairs},
		{&w.tiles, 16 * (pairs/16 + experts)},
		{&w.moeOut, 4 * pairOut},
		{&w.probeOut, 4 * maxProbe},
	} {
		if *b.dst, err = g.dev.Buffer(b.n); err != nil {
			return nil, err
		}
	}
	w.logitRows = 1
	eps := float32(c.eps)
	parts := uint32(c.hidden / rows)
	w.qkv0Args = gemvArgs{uint32(c.hidden), uint32(qdim + 2*c.kvDim), eps, 1}
	w.qkvArgs = gemvArgs{uint32(c.hidden), uint32(qdim + 2*c.kvDim), eps, parts}
	w.oArgs = gemvArgs{uint32(qdim), uint32(c.hidden), eps, 0}
	w.guArgs = gemvArgs{uint32(c.hidden), uint32(2 * c.intermediate), eps, parts}
	w.dArgs = gemvArgs{uint32(c.intermediate), uint32(c.hidden), eps, 0}
	w.headArgs = gemvArgs{uint32(c.hidden), uint32(g.lmRows), eps, 0}
	w.attn = attnArgs{ropeSin: uint32(g.positions * len(c.invFreq)), eps: eps, scale: float32(c.attnScale)}
	if c.experts > 0 {
		e, k := uint32(c.experts), uint32(c.topK)
		// A shared expert is expert E, in every token's last slot.
		sh := uint32(0)
		if c.hybrid {
			sh = 1
		}
		w.moeRouter = moeArgs{k: uint32(c.hidden), n: e, eps: eps, experts: e, topK: k, stride: uint32(routerRows(c)), shared: sh}
		w.moeGateUp = moeArgs{k: uint32(c.hidden), n: uint32(2 * c.intermediate), eps: eps, experts: e + sh, topK: k + sh}
		w.moeDown = moeArgs{k: uint32(c.intermediate), n: uint32(c.hidden), eps: eps, experts: e + sh, topK: k + sh, partsOut: uint32(c.hidden / rows)}
	}
	if c.hybrid {
		if err := g.newHybridWork(w); err != nil {
			return nil, err
		}
	}
	ok = true
	return w, nil
}

func (g *gpuModel) ensureDecodeVec() error {
	g.decodeVecOnce.Do(func() {
		discard := func() {
			for i := range g.decodeVec {
				for j, pipeline := range g.decodeVec[i] {
					pipeline.Release()
					g.decodeVec[i][j] = nil
				}
			}
			g.greedyArgmax.Release()
			g.greedyArgmax = nil
		}
		lib, err := g.dev.Compile(gpuDecodeBatchSourceFor(g.cfg))
		if err != nil {
			g.decodeVecErr = err
			return
		}
		defer lib.Release()
		projections := [...]string{"qkv", "o", "gateup", "down", "head"}
		widths := [...]int{1, 2, 4, 8}
		for wi, width := range widths {
			for pi, projection := range projections {
				name := fmt.Sprintf("gemv_vec_%s_m%d", projection, width)
				if g.decodeVec[wi][pi], err = g.dev.Pipeline(lib, name); err != nil {
					g.decodeVecErr = err
					discard()
					return
				}
			}
		}
		if g.greedyArgmax, err = g.dev.Pipeline(lib, "greedy_argmax_rows"); err != nil {
			g.decodeVecErr = err
			discard()
			return
		}
	})
	return g.decodeVecErr
}

func decodeVectorWidth(lanes int) int {
	// Capacity is limited to eight, so this short table is clearer than a
	// bit trick at the call site.
	switch {
	case lanes <= 1:
		return 1
	case lanes <= 2:
		return 2
	case lanes <= 4:
		return 4
	default:
		return 8
	}
}

func decodeVectorIndex(width int) int {
	switch width {
	case 1:
		return 0
	case 2:
		return 1
	case 4:
		return 2
	default:
		return 3
	}
}

func (m *Weights) newDecodeBatchWorkspace(capacity int) (*gpuDecodeBatchWorkspace, error) {
	g := m.gpu
	if g == nil || g.bits != 9 {
		return nil, errors.New("qwen3: batched token decode requires Q8B Metal weights")
	}
	if m.cfg.hybrid || m.cfg.experts != 0 {
		return nil, errors.New("qwen3: batched token decode supports dense attention models only")
	}
	m.headMu.Lock()
	headName := m.headName
	m.headMu.Unlock()
	if headName == "" {
		return nil, errors.New("qwen3: batched token decode requires a loaded language-model head; call LoadHead first")
	}
	if err := m.LoadHead(headName); err != nil {
		return nil, fmt.Errorf("qwen3: load batched decoder head: %w", err)
	}
	if !g.hasHead() {
		return nil, errors.New("qwen3: batched token decode requires a loaded language-model head")
	}
	if err := g.ensureDecodeVec(); err != nil {
		return nil, fmt.Errorf("qwen3: compile batched decode kernels: %w", err)
	}
	c, width := &m.cfg, decodeVectorWidth(capacity)
	qdim := c.heads * c.headDim
	qkvWidth := qdim + 2*c.kvDim
	parts := c.hidden / gpuRows(9)
	w := &gpuDecodeBatchWorkspace{g: g, capacity: capacity, vectorWidth: width,
		argmaxGPU: true,
		qkv0Args:  gemvArgs{k: uint32(c.hidden), n: uint32(qkvWidth), eps: float32(c.eps), parts: 1},
		qkvArgs:   gemvArgs{k: uint32(c.hidden), n: uint32(qkvWidth), eps: float32(c.eps), parts: uint32(parts)},
		oArgs:     gemvArgs{k: uint32(qdim), n: uint32(c.hidden), eps: float32(c.eps)},
		guArgs:    gemvArgs{k: uint32(c.hidden), n: uint32(2 * c.intermediate), eps: float32(c.eps), parts: uint32(parts)},
		dArgs:     gemvArgs{k: uint32(c.intermediate), n: uint32(c.hidden), eps: float32(c.eps)},
		headArgs:  gemvArgs{k: uint32(c.hidden), n: uint32(g.lmRows), eps: float32(c.eps)},
		attn:      attnArgs{ropeSin: uint32(g.positions * c.headDim / 2), eps: float32(c.eps), scale: float32(c.attnScale)},
		perRow:    uint32(c.intermediate / g.inter.block),
	}
	for _, buffer := range []struct {
		dst **metal.Buffer
		n   int
	}{
		{&w.h, 8 * width * c.hidden}, // input/residual rows plus final hidden staging
		{&w.qkv, 4 * width * qkvWidth},
		{&w.ctx, 4 * width * qdim},
		{&w.act, 4 * width * c.intermediate},
		{&w.logits, 4 * width * g.lmRows},
		{&w.embedParts, 4 * width},
		{&w.attnParts, 4 * width * parts},
		{&w.mlpParts, 4 * width * parts},
		{&w.tokenIDs, 4 * width},
	} {
		var err error
		if *buffer.dst, err = g.dev.Buffer(buffer.n); err != nil {
			w.release()
			return nil, err
		}
	}
	w.finalHidden = floats(w.h.Bytes())[width*c.hidden : 2*width*c.hidden]
	w.concurrent = true
	w.argmaxArgs = greedyArgmaxArgs{vocab: uint32(c.vocab), stride: uint32(g.lmRows)}
	return w, nil
}

func (w *gpuDecodeBatchWorkspace) release() {
	if w == nil {
		return
	}
	for _, buffer := range []*metal.Buffer{w.h, w.qkv, w.ctx, w.act, w.logits, w.tokenIDs, w.embedParts, w.attnParts, w.mlpParts} {
		buffer.Release()
	}
	w.h, w.qkv, w.ctx, w.act, w.logits, w.tokenIDs, w.embedParts, w.attnParts, w.mlpParts = nil, nil, nil, nil, nil, nil, nil, nil, nil
	w.finalHidden = nil
}

func (w *gpuDecodeBatchWorkspace) gemv(projection int, buf *metal.Buffer, wOff, sOff int,
	x, y, partsIn, partsOut *metal.Buffer, args *gemvArgs, head bool) {
	e := &w.enc
	e.SetPipeline(w.g.decodeVec[decodeVectorIndex(w.vectorWidth)][projection])
	e.SetBuffer(buf, wOff, 0)
	e.SetBuffer(buf, sOff, 1)
	e.SetBuffer(x, 0, 2)
	e.SetBuffer(y, 0, 3)
	e.SetBuffer(partsIn, 0, 4)
	e.SetBytes(unsafe.Pointer(args), 16, 5)
	e.SetBuffer(partsOut, 0, 6)
	rows := gpuRows(9)
	if head {
		rows = gpuHeadRows
	}
	e.Dispatch(metal.Size{X: int(args.n) / rows, Y: w.laneGroups, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
}

func (w *gpuDecodeBatchWorkspace) decode(m *Weights, kvs []*PrefixKV, tokens []int, hidden, logits [][]float32) error {
	return w.decodeMode(m, kvs, tokens, hidden, logits, w.concurrent)
}

func (w *gpuDecodeBatchWorkspace) decodeGreedy(m *Weights, kvs []*PrefixKV, tokens []int, hidden [][]float32, nextTokens []int) error {
	return w.decodeGreedyMode(m, kvs, tokens, hidden, nextTokens, w.concurrent, 4, w.argmaxGPU)
}

func (w *gpuDecodeBatchWorkspace) decodeGreedyMode(m *Weights, kvs []*PrefixKV, tokens []int, hidden [][]float32, nextTokens []int, concurrent bool, maxVectorWidth int, useGPUArgmax bool) error {
	return w.decodeModeTileOutputs(m, kvs, tokens, hidden, nil, nextTokens, concurrent, maxVectorWidth, useGPUArgmax)
}

func (w *gpuDecodeBatchWorkspace) decodeMode(m *Weights, kvs []*PrefixKV, tokens []int, hidden, logits [][]float32, concurrent bool) error {
	return w.decodeModeTile(m, kvs, tokens, hidden, logits, concurrent, 4)
}

// decodeModeTile permits a full-width M=8 control in tests and benchmarks;
// production uses M=4 tiles for eight active lanes to limit register pressure.
func (w *gpuDecodeBatchWorkspace) decodeModeTile(m *Weights, kvs []*PrefixKV, tokens []int, hidden, logits [][]float32, concurrent bool, maxVectorWidth int) error {
	return w.decodeModeTileOutputs(m, kvs, tokens, hidden, logits, nil, concurrent, maxVectorWidth, false)
}

func (w *gpuDecodeBatchWorkspace) decodeModeTileOutputs(m *Weights, kvs []*PrefixKV, tokens []int, hidden, logits [][]float32, nextTokens []int, concurrent bool, maxVectorWidth int, useGPUArgmax bool) error {
	g, c := w.g, &m.cfg
	active, qdim := len(kvs), c.heads*c.headDim
	width := decodeVectorWidth(active)
	if width > maxVectorWidth {
		width = maxVectorWidth
	}
	w.vectorWidth = width
	w.laneGroups = (active + width - 1) / width
	scratchWidth := decodeVectorWidth(w.capacity)
	allH := floats(w.h.Bytes())
	h := allH[:scratchWidth*c.hidden]
	clear(h)
	embedParts := floats(w.embedParts.Bytes())[:scratchWidth]
	clear(embedParts)
	for lane, token := range tokens {
		row := h[lane*c.hidden : (lane+1)*c.hidden]
		m.embedRow(token, row)
		g.hidden.apply(row)
		embedParts[lane] = sumSquares(row)
	}
	if scratchWidth > active {
		clear(floats(w.ctx.Bytes())[active*qdim : scratchWidth*qdim])
	}

	dev := g.dev
	dev.Begin(&w.enc, concurrent)
	barrier := func() {
		if concurrent {
			w.enc.Barrier()
		}
	}
	for layer := range g.layers {
		gl := &g.layers[layer]
		if layer == 0 {
			w.gemv(0, gl.buf, gl.qkv, gl.qkvScale, w.h, w.qkv, w.embedParts, w.attnParts, &w.qkv0Args, false)
		} else {
			w.gemv(0, gl.buf, gl.qkv, gl.qkvScale, w.h, w.qkv, w.attnParts, w.attnParts, &w.qkvArgs, false)
		}
		barrier() // QKV outputs feed every lane's independent attention dispatch.
		layerStride := 4 * c.kvDim
		for lane, kv := range kvs {
			a := w.attn
			a.pos = uint32(len(kv.tokens))
			e := &w.enc
			e.SetPipeline(g.attend)
			e.SetBuffer(w.qkv, lane*4*(qdim+2*c.kvDim), 0)
			e.SetBuffer(kv.gpu.kc, layer*layerStride*kv.capacity, 1)
			e.SetBuffer(kv.gpu.vc, layer*layerStride*kv.capacity, 2)
			e.SetBuffer(g.norms, 4*2*layer*c.headDim, 3)
			e.SetBuffer(g.norms, 4*(2*layer+1)*c.headDim, 4)
			e.SetBuffer(g.rope, 0, 5)
			e.SetBuffer(w.ctx, lane*4*qdim, 6)
			e.SetBytes(unsafe.Pointer(&a), int(unsafe.Sizeof(a)), 7)
			e.Dispatch(metal.Size{X: c.kvHeads, Y: 1, Z: 1}, metal.Size{X: 32 * attendSplits * c.heads / c.kvHeads, Y: 1, Z: 1})
		}
		barrier() // Lane attention writes ctx before the shared O projection.
		w.gemv(1, gl.buf, gl.o, gl.oScale, w.ctx, w.h, w.mlpParts, w.mlpParts, &w.oArgs, false)
		barrier()
		w.gemv(2, gl.buf, gl.gu, gl.guScale, w.h, w.act, w.mlpParts, w.mlpParts, &w.guArgs, false)
		barrier()
		if !g.noInter {
			w.enc.SetPipeline(g.rotate)
			w.enc.SetBuffer(w.act, 0, 0)
			w.enc.SetBuffer(g.signs, 0, 1)
			w.enc.SetBytes(unsafe.Pointer(&w.perRow), 4, 2)
			w.enc.Dispatch(metal.Size{X: active * int(w.perRow), Y: 1, Z: 1}, metal.Size{X: g.inter.block / 4, Y: 1, Z: 1})
			barrier()
		}
		w.gemv(3, gl.buf, gl.d, gl.dScale, w.act, w.h, w.attnParts, w.attnParts, &w.dArgs, false)
		barrier() // The next layer reads the residual and RMSNorm partials.
	}
	if err := w.enc.Wait(); err != nil {
		return err
	}

	for lane := range active {
		row := h[lane*c.hidden : (lane+1)*c.hidden]
		g.hidden.unapply(row)
		rmsNorm32(row, row, m.finalNorm, c.eps)
		for _, value := range row {
			if !finite32(value) {
				return errors.New("qwen3: non-finite batched hidden state")
			}
		}
		copy(w.finalHidden[lane*c.hidden:(lane+1)*c.hidden], row)
		g.hidden.apply(row)
	}

	dev.Begin(&w.enc, false)
	w.gemv(4, g.lm, 0, g.lmScale, w.h, w.logits, w.attnParts, w.mlpParts, &w.headArgs, true)
	if nextTokens != nil && useGPUArgmax {
		w.enc.SetPipeline(g.greedyArgmax)
		w.enc.SetBuffer(w.logits, 0, 0)
		w.enc.SetBuffer(w.tokenIDs, 0, 1)
		w.enc.SetBytes(unsafe.Pointer(&w.argmaxArgs), int(unsafe.Sizeof(w.argmaxArgs)), 2)
		w.enc.Dispatch(metal.Size{X: active, Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
	}
	if err := w.enc.Wait(); err != nil {
		return err
	}
	logitRows := floats(w.logits.Bytes())
	var tokenRows []uint32
	if nextTokens != nil && useGPUArgmax {
		tokenRows = unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(w.tokenIDs.Bytes()))), decodeVectorWidth(w.capacity))
	}
	for lane := range active {
		copy(hidden[lane], w.finalHidden[lane*c.hidden:(lane+1)*c.hidden])
		if logits != nil {
			copy(logits[lane], logitRows[lane*g.lmRows:lane*g.lmRows+c.vocab])
		}
		if nextTokens != nil {
			if useGPUArgmax {
				nextTokens[lane] = int(tokenRows[lane])
			} else {
				nextTokens[lane] = greedyArgmax(logitRows[lane*g.lmRows : lane*g.lmRows+c.vocab])
			}
		}
	}
	return nil
}

// encodeMoE encodes a layer's mixture of experts for rows rows of the
// residual at hOff: the router over the normalized residual (parts partial
// sums per row in mlpParts), each row's top experts, their SwiGLU, and the
// weighted sum of their down projections into the residual, which publishes
// hidden/16 partial sums per row in attnParts.
func (w *gpuWorkspace) encodeMoE(gl *gpuLayer, rows, hOff, parts int) {
	g, c, e := w.g, w.g.cfg, &w.enc
	w.moeRouter.parts, w.moeGateUp.parts = uint32(parts), uint32(parts)
	e.SetPipeline(g.moeRouter)
	e.SetBuffer(gl.buf, gl.router, 0)
	e.SetBuffer(w.h, hOff, 2)
	e.SetBuffer(w.route, 0, 3)
	e.SetBuffer(w.mlpParts, 0, 4)
	e.SetBytes(unsafe.Pointer(&w.moeRouter), int(unsafe.Sizeof(w.moeRouter)), 5)
	e.SetBuffer(w.counts, 0, 6)
	e.Dispatch(metal.Size{X: routerRows(c) / 16, Y: rows, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})

	e.SetPipeline(g.moeRoute)
	e.SetBuffer(w.route, 0, 0)
	e.SetBuffer(w.ids, 0, 1)
	e.SetBuffer(w.wts, 0, 2)
	e.SetBytes(unsafe.Pointer(&w.moeRouter), int(unsafe.Sizeof(w.moeRouter)), 3)
	e.SetBuffer(w.counts, 0, 4)
	e.SetBuffer(w.rank, 0, 5)
	e.Dispatch(metal.Size{X: rows, Y: 1, Z: 1}, metal.Size{X: c.experts, Y: 1, Z: 1})
	if rows >= moeGroupRows {
		w.encodeGroupedExperts(gl, rows, hOff)
		return
	}

	e.SetPipeline(g.moeGateUp)
	e.SetBuffer(gl.buf, gl.gu, 0)
	e.SetBuffer(gl.buf, gl.guScale, 1)
	e.SetBuffer(w.h, hOff, 2)
	e.SetBuffer(w.act, 0, 3)
	e.SetBuffer(w.mlpParts, 0, 4)
	e.SetBytes(unsafe.Pointer(&w.moeGateUp), int(unsafe.Sizeof(w.moeGateUp)), 5)
	e.SetBuffer(w.ids, 0, 6)
	e.Dispatch(metal.Size{X: 2 * c.intermediate / 16, Y: c.slots(), Z: rows}, metal.Size{X: gpuThreads, Y: 1, Z: 1})

	e.SetPipeline(g.moeDown)
	e.SetBuffer(gl.buf, gl.d, 0)
	e.SetBuffer(gl.buf, gl.dScale, 1)
	e.SetBuffer(w.act, 0, 2)
	e.SetBuffer(w.h, hOff, 3)
	e.SetBytes(unsafe.Pointer(&w.moeDown), int(unsafe.Sizeof(w.moeDown)), 5)
	e.SetBuffer(w.attnParts, 0, 6)
	e.SetBuffer(w.ids, 0, 7)
	e.SetBuffer(w.wts, 0, 8)
	e.Dispatch(metal.Size{X: c.hidden / 16, Y: rows, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
}

// encodeGroupedExperts encodes the experts of a batch grouped by expert:
// each expert multiplies all the rows routed to it as one GEMM, so its
// weights stream once per tile of pairs rather than once per row, and each
// row then sums its experts' weighted outputs into the residual.
func (w *gpuWorkspace) encodeGroupedExperts(gl *gpuLayer, rows, hOff int) {
	g, c, e := w.g, w.g.cfg, &w.enc
	pairs, experts := rows*c.slots(), c.allExperts()
	// Wide tiles once experts average 24 pairs or more.
	tile, kind := 16, 1
	if pairs >= 24*min(experts, pairs) {
		tile, kind = 32, 0
	}
	tiles := (pairs+tile-1)/tile + min(experts, pairs)
	w.moeTiles = moeTileArgs{pairs: uint32(pairs), tile: uint32(tile), tiles: uint32(tiles)}
	e.SetPipeline(g.moeTiles)
	e.SetBuffer(w.ids, 0, 0)
	e.SetBuffer(w.rank, 0, 1)
	e.SetBuffer(w.counts, 0, 2)
	e.SetBuffer(w.list, 0, 3)
	e.SetBuffer(w.tiles, 0, 4)
	e.SetBytes(unsafe.Pointer(&w.moeGateUp), int(unsafe.Sizeof(w.moeGateUp)), 5)
	e.SetBytes(unsafe.Pointer(&w.moeTiles), int(unsafe.Sizeof(w.moeTiles)), 6)
	e.Dispatch(metal.Size{X: 1, Y: 1, Z: 1}, metal.Size{X: 1024, Y: 1, Z: 1})

	e.SetPipeline(g.moeGateUpMM[kind])
	e.SetBuffer(gl.buf, gl.gu, 0)
	e.SetBuffer(gl.buf, gl.guScale, 1)
	e.SetBuffer(w.h, hOff, 2)
	e.SetBuffer(w.act, 0, 3)
	e.SetBuffer(w.mlpParts, 0, 4)
	e.SetBytes(unsafe.Pointer(&w.moeGateUp), int(unsafe.Sizeof(w.moeGateUp)), 5)
	e.SetBuffer(w.list, 0, 6)
	e.SetBuffer(w.tiles, 0, 7)
	e.SetBuffer(w.wts, 0, 8)
	e.Dispatch(metal.Size{X: 2 * c.intermediate / mmColumns, Y: tiles, Z: 1}, metal.Size{X: mmThreads, Y: 1, Z: 1})

	e.SetPipeline(g.moeDownMM[kind])
	e.SetBuffer(gl.buf, gl.d, 0)
	e.SetBuffer(gl.buf, gl.dScale, 1)
	e.SetBuffer(w.act, 0, 2)
	e.SetBuffer(w.moeOut, 0, 3)
	e.SetBytes(unsafe.Pointer(&w.moeDown), int(unsafe.Sizeof(w.moeDown)), 5)
	e.Dispatch(metal.Size{X: c.hidden / mmColumns, Y: tiles, Z: 1}, metal.Size{X: mmThreads, Y: 1, Z: 1})

	e.SetPipeline(g.moeCombine)
	e.SetBuffer(w.moeOut, 0, 0)
	e.SetBuffer(w.h, hOff, 3)
	e.SetBuffer(w.attnParts, 0, 6)
	e.Dispatch(metal.Size{X: c.hidden / mmColumns, Y: rows, Z: 1}, metal.Size{X: mmColumns, Y: 1, Z: 1})
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
	rows := gpuRows(w.g.bits)
	if p == w.g.gemvHead {
		rows = gpuHeadRows
	}
	e.Dispatch(metal.Size{X: int(args.n) / rows, Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
}

// batch evaluates independent sequences and writes each one's last-token
// post-final-norm state to dst. Sequences are packed into shared forward
// passes of up to gpuPositions rows; a lone single token takes the GEMV path.
// With a prefix pre of past tokens, sequences continue it: shared prefixes
// are only read, and an unshared one (a single sequence) receives the new
// keys and values after its first past rows, a pass at a time when it is
// longer than one pass. Placeholder tokens of embeds take its rows.
func (w *gpuWorkspace) batch(m *Weights, seqs [][]int, dst [][]float32, pre *gpuPrefix, past int, shared bool, embeds Embeds) error {
	if !m.cfg.hybrid {
		return w.batchPacked(m, seqs, dst, pre, past, shared, embeds)
	}
	// A Qwen3.5 sequence runs on its own recurrent state: the prefix's when
	// it extends one, a copy of it for each continuation of a shared prefix,
	// or the workspace's from nothing.
	if pre != nil && !shared && len(seqs) > 1 {
		return errors.New("qwen3: a Qwen3.5 model extends one sequence at a time")
	}
	for s := range seqs {
		state := w.hy.own
		switch {
		case shared:
			state.copyFrom(pre.state)
		case pre != nil:
			state = pre.state
			if past == 0 {
				state.reset()
			}
		default:
			state.reset()
		}
		w.hy.state = state
		if err := w.batchPacked(m, seqs[s:s+1], dst[s:s+1], pre, past, shared, embeds); err != nil {
			return err
		}
	}
	return nil
}

// batchPacked is batch with sequences packed into shared passes.
func (w *gpuWorkspace) batchPacked(m *Weights, seqs [][]int, dst [][]float32, pre *gpuPrefix, past int, shared bool, embeds Embeds) error {
	c := &m.cfg
	w.attn.base, w.attn.prefixLen = 0, 0
	if pre != nil && !shared {
		// Continue in the lane-owned cache. The prefix input has length zero;
		// binding it to the same valid buffer keeps both attention passes safe
		// without allocating the workspace's otherwise-unused full cache.
		stride := 4 * pre.capacity * c.kvDim
		w.curK, w.curV, w.curStride = pre.kc, pre.vc, stride
		w.preK, w.preV, w.preStride = pre.kc, pre.vc, stride
		w.attn.base = uint32(past)
	} else {
		if err := w.ensureKV(); err != nil {
			return err
		}
		own := 4 * gpuPositions * c.kvDim
		w.curK, w.curV, w.curStride = w.kc, w.vc, own
		w.preK, w.preV, w.preStride = w.kc, w.vc, own
		if pre != nil {
			w.preK, w.preV = pre.kc, pre.vc
			w.preStride = 4 * pre.capacity * c.kvDim
			w.attn.prefixLen = uint32(past)
		}
	}
	w.past, w.shared, w.embeds, w.spliced = past, shared, embeds, 0
	defer func() { w.embeds, w.oneSeq[0] = Embeds{}, nil }()
	if pre != nil && !shared && len(seqs) == 1 && len(seqs[0]) > w.rows {
		ids := seqs[0]
		k := len(w.tail) / c.hidden
		for off := 0; off < len(ids); off += len(w.oneSeq[0]) {
			n := min(w.rows, len(ids)-off)
			if rest := len(ids) - off - n; rest > 0 && rest < k {
				n = len(ids) - off - k // the last pass holds the whole tail
			}
			w.oneSeq[0] = ids[off : off+n]
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

// logitsRowsInto writes the head's logits for k post-final-norm states, all
// in one command buffer.
func (w *gpuWorkspace) logitsRowsInto(m *Weights, hidden, dst []float32, k int) error {
	g, c := w.g, &m.cfg
	if err := w.fitHead(k); err != nil {
		return err
	}
	x := floats(w.h.Bytes())[:k*c.hidden]
	copy(x, hidden)
	for r := range k {
		g.hidden.apply(x[r*c.hidden : (r+1)*c.hidden])
	}
	e := &w.enc
	g.dev.Begin(e, false)
	if g.lmRows%mmColumns == 0 {
		// One GEMM reads the head once for every row; its grid is wide
		// enough that it never splits K.
		w.mmRun(g.mmHead[0], g.mmHead[1], g.finishHead, g.lm, 0, g.lmScale, w.h, w.logits, w.attnParts, w.attnParts, c.hidden, g.lmRows, k, 0)
	} else {
		for r := range k {
			w.gemv(g.gemvHead, g.lm, 0, g.lmScale, w.h, 4*r*c.hidden, w.logits, 4*r*g.lmRows, w.attnParts, w.attnParts, 0, &w.headArgs)
		}
	}
	if err := e.Wait(); err != nil {
		return err
	}
	out := floats(w.logits.Bytes())
	for r := range k {
		copy(dst[r*c.vocab:(r+1)*c.vocab], out[r*g.lmRows:r*g.lmRows+c.vocab])
	}
	return nil
}

// fitHead sizes the logits buffer for k rows of the head, which may have
// loaded after the workspace was made.
func (w *gpuWorkspace) fitHead(k int) error {
	g := w.g
	w.headArgs.n = uint32(g.lmRows)
	if len(w.logits.Bytes()) >= 4*k*g.lmRows {
		return nil
	}
	b, err := g.dev.Buffer(4 * max(k, w.logitRows) * g.lmRows)
	if err != nil {
		return err
	}
	w.logits.Release()
	w.logits, w.logitRows = b, max(k, w.logitRows)
	return nil
}

// logits writes the head's logits for one post-final-norm state.
func (w *gpuWorkspace) logitsInto(m *Weights, hidden, dst []float32) error {
	g, c := w.g, &m.cfg
	if err := w.fitHead(1); err != nil {
		return err
	}
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
	hybrid := c.hybrid
	if hybrid {
		// A hybrid pass is one sequence, continuing the state it runs on.
		if len(seqs) != 1 || w.hy.state.pos != w.past {
			return fmt.Errorf("qwen3: a Qwen3.5 pass continues its state at %d tokens, not %d", w.hy.state.pos, w.past)
		}
	}
	switch {
	case (gpuTokenByToken || rows == 1) && !w.shared:
		if len(seqs) != 1 {
			panic("qwen3: token-by-token GPU path takes one sequence")
		}
		if hybrid {
			w.encodeHybridTokens(rows)
		} else {
			w.encodeTokens(rows)
		}
	case hybrid:
		w.encodeHybridBatch(rows)
	default:
		w.encodeBatch(rows)
	}
	if err := e.Wait(); err != nil {
		return err
	}
	if hybrid {
		w.hy.state.pos += rows
	}
	if p := w.probe; p != nil {
		copy(p.Probs, floats(w.probeOut.Bytes()))
	}
	if k := len(w.tail) / c.hidden; k > 1 && len(seqs) == 1 && rows >= k {
		for i := range k {
			out := w.tail[i*c.hidden : (i+1)*c.hidden]
			copy(out, hs[(rows-k+i)*c.hidden:(rows-k+i+1)*c.hidden])
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
	w.mmRun(w.g.mm[0][kind], w.g.mm[1][kind], w.g.finish[[4]int{0, 1, 2, 1}[kind]], buf, wOff, sOff, x, y, in, out, k, n, rows, parts)
}

// mmRun encodes one GEMM with the given 32- and 16-token tile pipelines and
// split-K epilogue.
func (w *gpuWorkspace) mmRun(p32, p16, finish *metal.Pipeline, buf *metal.Buffer, wOff, sOff int, x, y, in, out *metal.Buffer,
	k, n, rows, parts int) {
	e := &w.enc
	// Up to 16 tokens use the 16-token tile, which wastes no rows.
	tile, p := 32, p32
	if rows <= 16 {
		tile, p = 16, p16
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
	e.Dispatch(metal.Size{X: n / mmColumns, Y: (rows + tile - 1) / tile, Z: splits}, metal.Size{X: mmThreads, Y: 1, Z: 1})
	if splits > 1 {
		e.SetPipeline(finish)
		e.Dispatch(metal.Size{X: n / mmColumns, Y: rows, Z: 1}, metal.Size{X: mmColumns, Y: 1, Z: 1})
	}
}

// encodeBatch encodes a forward pass over rows tokens at positions 0..rows-1
// with batched projections, reading each weight once per 32 tokens.
func (w *gpuWorkspace) encodeBatch(rows int) {
	g, c, e := w.g, w.g.cfg, &w.enc
	qdim := c.heads * c.headDim
	parts := c.hidden / mmColumns
	// The partial sums the next layer's QKV reads: the down projection's,
	// or the experts' (one per 16 rows).
	qkvParts := parts
	if c.experts > 0 {
		qkvParts = c.hidden / 16
	}
	w.attn.pos = 0
	w.perRow = uint32(c.intermediate / g.inter.block)
	for i := range g.layers {
		gl := &g.layers[i]
		if i == 0 {
			w.mmDispatch(0, gl.buf, gl.qkv, gl.qkvScale, w.h, w.qkv, w.embedParts, w.attnParts, c.hidden, qdim+2*c.kvDim, rows, 1)
		} else {
			w.mmDispatch(0, gl.buf, gl.qkv, gl.qkvScale, w.h, w.qkv, w.attnParts, w.attnParts, c.hidden, qdim+2*c.kvDim, rows, qkvParts)
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
		if c.experts > 0 {
			w.encodeMoE(gl, rows, 0, parts)
			continue
		}
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
			if p := w.probe; p != nil && p.Layer == i && t == n-1 {
				w.encodeProbe(i)
			}

			w.gemv(g.o, gl.buf, gl.o, gl.oScale, w.ctx, 0, w.h, hOff, w.mlpParts, w.mlpParts, 0, &w.oArgs)
			if c.experts > 0 {
				w.encodeMoE(gl, 1, hOff, c.hidden/gpuRows(g.bits))
				continue
			}
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

// gpuDecoder is a Decoder's heads and tables in GPU memory, FP16: heads
// [steps][padded][hidden], with the final norm's weight and the rotation
// folded in; tables [tables][rows][hidden], rotated, with each row's sum of
// squares. It is read-only: a run's tokens and logits are its workspace's.
type gpuDecoder struct {
	heads, tables, sumsq *metal.Buffer
	rows, padded         int
}

// decodeArgs is DecodeArgs in gpu.metal.
type decodeArgs struct {
	k, rows, parts uint32
	eps            float32
	topK           uint32
	invTemp        float32
	seed, draw     [2]uint32
	step           uint32
	_              uint32
}

func (g *gpuModel) newDecoder(heads, tables [][]float32, rows int, finalNorm []float32) (*gpuDecoder, error) {
	h := g.cfg.hidden
	d := &gpuDecoder{rows: rows, padded: (rows + 31) / 32 * 32}
	var err error
	if d.heads, err = g.dev.Buffer(2 * len(heads) * d.padded * h); err != nil {
		return nil, err
	}
	halves := func(b *metal.Buffer) []uint16 {
		if b == nil {
			return nil
		}
		return unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(b.Bytes()))), len(b.Bytes())/2)
	}
	// A single-step decoder never dispatches decodeGather. No input table
	// or sum-of-squares buffer is needed, even for a large vocabulary.
	if len(tables) != 0 {
		if d.tables, err = g.dev.Buffer(2 * len(tables) * rows * h); err != nil {
			d.release()
			return nil, err
		}
		if d.sumsq, err = g.dev.Buffer(4 * len(tables) * rows); err != nil {
			d.release()
			return nil, err
		}
	}
	hw, tw := halves(d.heads), halves(d.tables)
	var sq []float32
	if d.sumsq != nil {
		sq = floats(d.sumsq.Bytes())
	}
	// Rows are rotated as the residual is, a matrix to a goroutine.
	var wg sync.WaitGroup
	prepare := func(src []float32, dst []uint16, norm []float32, sums []float32) {
		defer wg.Done()
		row := make([]float32, h)
		for r := range rows {
			copy(row, src[r*h:(r+1)*h])
			if norm != nil {
				for i := range row {
					row[i] *= norm[i]
				}
			}
			g.hidden.apply(row)
			var ss float32
			for i, v := range row {
				dst[r*h+i] = q8gemm.F32ToF16(v)
				v = safetensors.F16ToF32(dst[r*h+i])
				ss += v * v
			}
			if sums != nil {
				sums[r] = ss
			}
		}
	}
	for i, m := range heads {
		wg.Add(1)
		go prepare(m, hw[i*d.padded*h:], finalNorm, nil)
	}
	for i, m := range tables {
		wg.Add(1)
		go prepare(m, tw[i*rows*h:], nil, sq[i*rows:])
	}
	wg.Wait()
	return d, nil
}

func (d *gpuDecoder) release() {
	for _, b := range []*metal.Buffer{d.heads, d.tables, d.sumsq} {
		if b != nil {
			b.Release()
		}
	}
}

// decode encodes a Decoder's run in one command buffer: the input rows as
// a pass does, then for each step the head over the last state, the draw,
// and, but for the last, the drawn token's row as a one-token pass.
func (w *gpuWorkspace) decode(m *Weights, d *gpuDecoder, pre *gpuPrefix, past int, ids []int, embeds Embeds, s Sampling, tokens []int, logits []float32) error {
	g, c, e := w.g, &m.cfg, &w.enc
	h := c.hidden
	if len(ids) > w.rows {
		return fmt.Errorf("qwen3: %d input tokens exceed the GPU context %d", len(ids), w.rows)
	}
	for _, b := range []struct {
		buf **metal.Buffer
		n   int
	}{{&w.decodeTokens, 4 * len(tokens)}, {&w.decodeLogits, 4 * len(tokens) * d.padded}} {
		if *b.buf == nil || len((*b.buf).Bytes()) < b.n {
			nb, err := g.dev.Buffer(b.n)
			if err != nil {
				return err
			}
			if *b.buf != nil {
				(*b.buf).Release()
			}
			*b.buf = nb
		}
	}
	stride := 4 * pre.capacity * c.kvDim
	w.curK, w.curV, w.curStride = pre.kc, pre.vc, stride
	w.preK, w.preV, w.preStride = pre.kc, pre.vc, stride
	w.attn.base, w.attn.prefixLen = uint32(past), 0
	w.past, w.shared = past, false
	hs, parts := floats(w.h.Bytes()), floats(w.embedParts.Bytes())
	spliced := 0
	for t, id := range ids {
		row := hs[t*h : (t+1)*h]
		if len(embeds.Rows) != 0 && id == embeds.Token {
			copy(row, embeds.Rows[spliced*h:])
			spliced++
		} else {
			m.embedRow(id, row)
		}
		g.hidden.apply(row)
		parts[t] = sumSquares(row)
	}
	g.dev.Begin(e, false)
	w.encodeTokens(len(ids))
	args := &w.decodeArgs
	*args = decodeArgs{k: uint32(h), rows: uint32(d.rows), parts: w.qkvArgs.parts, eps: w.qkvArgs.eps,
		topK: uint32(s.TopK), invTemp: 1 / s.Temperature, seed: [2]uint32{uint32(s.Seed), uint32(s.Seed >> 32)}}
	size := int(unsafe.Sizeof(*args))
	last := len(ids) - 1 // the row of the state the next head reads
	for i := range tokens {
		draw := s.Draw + uint64(i)
		args.step, args.draw = uint32(i), [2]uint32{uint32(draw), uint32(draw >> 32)}
		e.SetPipeline(g.decodeHead)
		e.SetBuffer(d.heads, 2*i*d.padded*h, 0)
		e.SetBuffer(w.h, 4*last*h, 1)
		e.SetBuffer(w.attnParts, 0, 2)
		e.SetBuffer(w.decodeLogits, 4*i*d.padded, 3)
		e.SetBytes(unsafe.Pointer(args), size, 4)
		e.Dispatch(metal.Size{X: d.padded / 32, Y: 1, Z: 1}, metal.Size{X: 256, Y: 1, Z: 1})

		e.SetPipeline(g.decodeSample)
		e.SetBuffer(w.decodeLogits, 4*i*d.padded, 0)
		e.SetBuffer(w.decodeTokens, 0, 1)
		e.SetBytes(unsafe.Pointer(args), size, 4)
		e.Dispatch(metal.Size{X: 1, Y: 1, Z: 1}, metal.Size{X: 1024, Y: 1, Z: 1})
		if i == len(tokens)-1 {
			break
		}
		e.SetPipeline(g.decodeGather)
		e.SetBuffer(d.tables, 2*i*d.rows*h, 0)
		e.SetBuffer(d.sumsq, 4*i*d.rows, 1)
		e.SetBuffer(w.decodeTokens, 0, 2)
		e.SetBuffer(w.h, 0, 3)
		e.SetBuffer(w.embedParts, 0, 4)
		e.SetBytes(unsafe.Pointer(args), size, 5)
		e.Dispatch(metal.Size{X: 1, Y: 1, Z: 1}, metal.Size{X: 256, Y: 1, Z: 1})
		w.past += last + 1
		last = 0
		w.encodeTokens(1)
	}
	if err := e.Wait(); err != nil {
		return err
	}
	out := unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(w.decodeTokens.Bytes()))), len(tokens))
	for i, t := range out {
		tokens[i] = int(t)
	}
	if logits != nil {
		all := floats(w.decodeLogits.Bytes())
		for i := range tokens {
			copy(logits[i*d.rows:(i+1)*d.rows], all[i*d.padded:])
		}
	}
	return nil
}

// encodeProbe reads w.probe's head of layer i as attend1 weighed it, before
// the next layer overwrites the query.
func (w *gpuWorkspace) encodeProbe(i int) {
	g, c, e, p := w.g, w.g.cfg, &w.enc, w.probe
	w.probeArgs = probeArgs{uint32(p.Head), uint32(p.From), uint32(len(p.Probs))}
	e.SetPipeline(g.probe)
	e.SetBuffer(w.qkv, 0, 0)
	e.SetBuffer(w.curK, i*w.curStride, 1)
	e.SetBuffer(g.norms, 4*2*i*c.headDim, 3)
	e.SetBuffer(g.rope, 0, 5)
	e.SetBuffer(w.probeOut, 0, 6)
	e.SetBytes(unsafe.Pointer(&w.attn), int(unsafe.Sizeof(w.attn)), 7)
	e.SetBytes(unsafe.Pointer(&w.probeArgs), int(unsafe.Sizeof(w.probeArgs)), 8)
	e.Dispatch(metal.Size{X: 1, Y: 1, Z: 1}, metal.Size{X: 32 * attendSplits, Y: 1, Z: 1})
}

func (w *gpuWorkspace) release() {
	for _, b := range []*metal.Buffer{w.h, w.qkv, w.ctx, w.act, w.kc, w.vc, w.embedParts, w.info, w.scratch, w.attnParts, w.mlpParts, w.logits, w.route, w.ids, w.wts, w.counts, w.rank, w.list, w.tiles, w.moeOut, w.probeOut} {
		b.Release()
	}
	for _, b := range []*metal.Buffer{w.decodeTokens, w.decodeLogits} {
		if b != nil {
			b.Release()
		}
	}
	w.hy.release()
}

func (m *Weights) releaseGPU() {
	g := m.gpu
	if g == nil {
		return
	}
	for i := range g.layers {
		g.layers[i].buf.Release()
	}
	for _, p := range []*metal.Pipeline{
		g.qkv, g.o, g.gateup, g.down, g.attend, g.probe, g.decodeHead, g.decodeSample, g.decodeGather,
		g.rotate, g.qkRope, g.attendM, g.attendFlash, g.gemvHead, g.finishHead,
		g.moeRouter, g.moeRoute, g.moeGateUp, g.moeDown, g.moeTiles, g.moeCombine,
	} {
		p.Release()
	}
	for _, p := range g.decodeVec {
		for _, pipeline := range p {
			pipeline.Release()
		}
	}
	g.greedyArgmax.Release()
	for _, p := range g.mm {
		for _, pipeline := range p {
			pipeline.Release()
		}
	}
	for _, p := range g.finish {
		p.Release()
	}
	for _, p := range g.mmHead {
		p.Release()
	}
	for _, p := range g.moeGateUpMM {
		p.Release()
	}
	for _, p := range g.moeDownMM {
		p.Release()
	}
	g.hy.release()
	for _, b := range []*metal.Buffer{g.lm, g.norms, g.signs, g.rope, g.dnParams} {
		b.Release()
	}
	g.cache.Close()
	g.headCache.Close()
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

// cacheKind names the kind of a cache entry of m's weights, suffix added:
// the format, and the decoder's tensor prefix when the checkpoint holds
// more than one decoder (Qwen3-TTS's talker and code predictor), since an
// entry replaces the others of its checkpoint and kind when it is made.
func (m *Weights) cacheKind(suffix string) string {
	switch m.prefix {
	case "", "model.":
		return m.format + suffix
	}
	return m.format + suffix + "-" + strings.TrimSuffix(m.prefix, ".")
}
