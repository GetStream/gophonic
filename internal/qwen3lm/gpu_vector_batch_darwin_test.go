// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"testing"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/thesyncim/vibejson"
)

func batchBenchBuffer(b *testing.B, dev *metal.Device, size int) *metal.Buffer {
	b.Helper()
	buffer, err := dev.Buffer(size)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(buffer.Release)
	return buffer
}

func fillBenchRows(dst []float32, rows, width int, rng *rand.Rand) []float32 {
	sums := make([]float32, rows)
	for row := range rows {
		values := dst[row*width : (row+1)*width]
		for i := range values {
			v := float32(rng.NormFloat64()) * 0.02
			values[i] = v
			sums[row] += v * v
		}
	}
	return sums
}

func fillBenchNormParts(dst, sums []float32, parts int) {
	clear(dst)
	for row, sum := range sums {
		if parts > 0 {
			dst[row*parts] = sum
		}
	}
}

// TestGPUQ8BVectorGemvParity checks the multi-vector kernels against M
// independent FP32 GEMVs for every projection epilogue, including the fixed
// head geometry. The kernels consume the same stored Q8B bytes and scales.
func TestGPUQ8BVectorGemvParity(t *testing.T) {
	const (
		vecPlain = iota
		vecNorm
	)
	const (
		vecStore = iota
		vecAdd
		vecSwiGLU
	)
	dev, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	c := &modelConfig{hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, kvDim: 1024, intermediate: 6144}
	lib, err := dev.Compile(gpuDecodeBatchSourceFor(c))
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Release()
	projections := []struct {
		name, serial, vector string
		pro, epi             int
		head                 bool
	}{
		{"qkv", "gemv_qkv_q8", "gemv_vec_qkv_m", vecNorm, vecStore, false},
		{"o", "gemv_o_q8", "gemv_vec_o_m", vecPlain, vecAdd, false},
		{"gateup", "gemv_gateup_q8", "gemv_vec_gateup_m", vecNorm, vecSwiGLU, false},
		{"down", "gemv_down_q8", "gemv_vec_down_m", vecPlain, vecAdd, false},
		{"head", "gemv_head", "gemv_vec_head_m", vecPlain, vecStore, true},
	}
	const k, n = 2048, 128
	devW, err := dev.Buffer(n * k)
	if err != nil {
		t.Fatal(err)
	}
	defer devW.Release()
	devScale, err := dev.Buffer(2 * n * k / q4Group)
	if err != nil {
		t.Fatal(err)
	}
	defer devScale.Release()
	rng := rand.New(rand.NewPCG(311, 821))
	weights := unsafe.Slice((*int8)(unsafe.Pointer(&devW.Bytes()[0])), n*k)
	scales := unsafe.Slice((*uint16)(unsafe.Pointer(&devScale.Bytes()[0])), n*k/q4Group)
	rowWeights := make([]float32, k)
	for row := range n {
		for i := range rowWeights {
			rowWeights[i] = float32(rng.NormFloat64()) * 0.02
		}
		quantizeRowQ8B(rowWeights, weights[row*k:(row+1)*k], scales[row*k/q4Group:(row+1)*k/q4Group], 1)
	}

	for _, projection := range projections {
		serial, err := dev.Pipeline(lib, projection.serial)
		if err != nil {
			t.Fatal(err)
		}
		defer serial.Release()
		for _, lanes := range []int{1, 2, 4, 8} {
			vector, err := dev.Pipeline(lib, fmt.Sprintf("%s%d", projection.vector, lanes))
			if err != nil {
				t.Fatal(err)
			}
			defer vector.Release()
			t.Run(fmt.Sprintf("%s/M=%d", projection.name, lanes), func(t *testing.T) {
				rows := gpuRows(9)
				if projection.head {
					rows = gpuHeadRows
				}
				parts := 0
				if projection.pro == vecNorm {
					parts = k / rows
				}
				args := gemvArgs{k: k, n: n, eps: 1e-6, parts: uint32(parts)}
				x, err := dev.Buffer(4 * lanes * k)
				if err != nil {
					t.Fatal(err)
				}
				defer x.Release()
				inParts, err := dev.Buffer(4 * max(lanes*parts, 1))
				if err != nil {
					t.Fatal(err)
				}
				defer inParts.Release()
				groups := n / rows
				outParts, err := dev.Buffer(4 * lanes * groups)
				if err != nil {
					t.Fatal(err)
				}
				defer outParts.Release()
				outN := n
				if projection.epi == vecSwiGLU {
					outN /= 2
				}
				ySerial, err := dev.Buffer(4 * lanes * outN)
				if err != nil {
					t.Fatal(err)
				}
				defer ySerial.Release()
				yVector, err := dev.Buffer(4 * lanes * outN)
				if err != nil {
					t.Fatal(err)
				}
				defer yVector.Release()
				xs := floats(x.Bytes())
				var sums []float32
				if projection.pro == vecNorm {
					sums = fillBenchRows(xs, lanes, k, rng)
				} else {
					fillBenchRows(xs, lanes, k, rng)
				}
				if parts > 0 {
					fillBenchNormParts(floats(inParts.Bytes()), sums, parts)
				}
				var initial []float32
				if projection.epi == vecAdd {
					initial = make([]float32, lanes*outN)
					for i := range initial {
						initial[i] = float32(rng.NormFloat64()) * 0.01
					}
					copy(floats(ySerial.Bytes()), initial)
					copy(floats(yVector.Bytes()), initial)
				}

				var e metal.Encoder
				dev.Begin(&e, false)
				for lane := range lanes {
					inputOffset, outputOffset := 4*lane*k, 4*lane*outN
					inOffset := 4 * lane * parts
					outOffset := 4 * lane * groups
					e.SetPipeline(serial)
					e.SetBuffer(devW, 0, 0)
					e.SetBuffer(devScale, 0, 1)
					e.SetBuffer(x, inputOffset, 2)
					e.SetBuffer(ySerial, outputOffset, 3)
					e.SetBuffer(inParts, inOffset, 4)
					e.SetBytes(unsafe.Pointer(&args), 16, 5)
					e.SetBuffer(outParts, outOffset, 6)
					e.Dispatch(metal.Size{X: n / rows, Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
				}
				if err := e.Wait(); err != nil {
					t.Fatal(err)
				}

				dev.Begin(&e, false)
				e.SetPipeline(vector)
				e.SetBuffer(devW, 0, 0)
				e.SetBuffer(devScale, 0, 1)
				e.SetBuffer(x, 0, 2)
				e.SetBuffer(yVector, 0, 3)
				e.SetBuffer(inParts, 0, 4)
				e.SetBytes(unsafe.Pointer(&args), 16, 5)
				e.SetBuffer(outParts, 0, 6)
				e.Dispatch(metal.Size{X: n / rows, Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
				if err := e.Wait(); err != nil {
					t.Fatal(err)
				}
				want, got := floats(ySerial.Bytes()), floats(yVector.Bytes())
				maxAbs, rel, finite := gpuVectorError(want, got)
				if !finite || maxAbs > 1e-4 || rel > 1e-5 {
					t.Fatalf("non-parity output: max_abs %.3g relative %.3g finite=%t", maxAbs, rel, finite)
				}
				if lanes == 8 {
					tiled, err := dev.Pipeline(lib, fmt.Sprintf("%s4", projection.vector))
					if err != nil {
						t.Fatal(err)
					}
					defer tiled.Release()
					if projection.epi == vecAdd {
						copy(floats(yVector.Bytes()), initial)
					}
					dev.Begin(&e, false)
					e.SetPipeline(tiled)
					e.SetBuffer(devW, 0, 0)
					e.SetBuffer(devScale, 0, 1)
					e.SetBuffer(x, 0, 2)
					e.SetBuffer(yVector, 0, 3)
					e.SetBuffer(inParts, 0, 4)
					e.SetBytes(unsafe.Pointer(&args), 16, 5)
					e.SetBuffer(outParts, 0, 6)
					e.Dispatch(metal.Size{X: n / rows, Y: 2, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
					if err := e.Wait(); err != nil {
						t.Fatal(err)
					}
					maxAbs, rel, finite = gpuVectorError(floats(ySerial.Bytes()), floats(yVector.Bytes()))
					if !finite || maxAbs > 1e-4 || rel > 1e-5 {
						t.Fatalf("M4x2 parity: max_abs %.3g relative %.3g finite=%t", maxAbs, rel, finite)
					}
				}
			})
		}
	}
}

func TestGPUGreedyArgmaxStable(t *testing.T) {
	dev, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	c := &modelConfig{hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, intermediate: 6144}
	lib, err := dev.Compile(gpuDecodeBatchSourceFor(c))
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Release()
	pipeline, err := dev.Pipeline(lib, "greedy_argmax_rows")
	if err != nil {
		t.Fatal(err)
	}
	defer pipeline.Release()

	const rows, vocab, stride = 10, 257, 260
	input, err := dev.Buffer(4 * rows * stride)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Release()
	output, err := dev.Buffer(4 * rows)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Release()
	values := floats(input.Bytes())
	for row := range rows {
		rowValues := values[row*stride : (row+1)*stride]
		for i := range vocab {
			rowValues[i] = float32(i) * -1
		}
		// Padded entries must never participate in the argmax.
		for i := vocab; i < stride; i++ {
			rowValues[i] = float32(math.Inf(1))
		}
	}
	cases := []struct {
		name string
		set  func([]float32)
	}{
		{"first-tie", func(v []float32) { v[7], v[19] = 42, 42 }},
		{"positive-infinity-tie", func(v []float32) { v[2], v[11] = float32(math.Inf(1)), float32(math.Inf(1)) }},
		{"all-negative-infinity", func(v []float32) {
			clear(v[:vocab])
			for i := range vocab {
				v[i] = float32(math.Inf(-1))
			}
		}},
		{"nan-at-zero", func(v []float32) { v[0], v[2] = float32(math.NaN()), 99 }},
		{"later-nan-ignored", func(v []float32) { v[1], v[8], v[9] = 20, float32(math.NaN()), 19 }},
		{"signed-zero-first-negative", func(v []float32) { v[0], v[4] = float32(math.Copysign(0, -1)), 0 }},
		{"signed-zero-first-positive", func(v []float32) { v[0], v[4] = 0, float32(math.Copysign(0, -1)) }},
		{"positive-infinity-wins", func(v []float32) { v[1], v[200] = float32(math.Inf(-1)), float32(math.Inf(1)) }},
		{"last-valid-index", func(v []float32) { v[256] = 1 }},
		{"subnormal-positive-wins", func(v []float32) { v[0], v[1] = math.Float32frombits(0x80000001), math.Float32frombits(1) }},
	}
	for row, test := range cases {
		rowValues := values[row*stride : (row+1)*stride]
		clear(rowValues[:vocab])
		test.set(rowValues)
	}
	args := greedyArgmaxArgs{vocab: vocab, stride: stride}
	var enc metal.Encoder
	dev.Begin(&enc, false)
	enc.SetPipeline(pipeline)
	enc.SetBuffer(input, 0, 0)
	enc.SetBuffer(output, 0, 1)
	enc.SetBytes(unsafe.Pointer(&args), 8, 2)
	enc.Dispatch(metal.Size{X: rows, Y: 1, Z: 1}, metal.Size{X: 256, Y: 1, Z: 1})
	if err := enc.Wait(); err != nil {
		t.Fatal(err)
	}
	got := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(output.Bytes()))), rows)
	for row, test := range cases {
		want := greedyArgmax(values[row*stride : row*stride+vocab])
		if int(got[row]) != want {
			t.Errorf("%s: GPU token %d, CPU token %d", test.name, got[row], want)
		}
	}
}

func gpuVectorError(want, got []float32) (maxAbs, rel float64, finite bool) {
	finite = len(want) == len(got)
	if !finite {
		return 0, math.Inf(1), false
	}
	var e2, w2 float64
	for i, a := range want {
		b := got[i]
		if math.IsNaN(float64(a)) || math.IsInf(float64(a), 0) || math.IsNaN(float64(b)) || math.IsInf(float64(b), 0) {
			finite = false
			continue
		}
		d := float64(a - b)
		maxAbs = math.Max(maxAbs, math.Abs(d))
		e2 += d * d
		w2 += float64(a) * float64(a)
	}
	if w2 == 0 {
		rel = math.Sqrt(e2)
	} else {
		rel = math.Sqrt(e2 / w2)
	}
	return maxAbs, rel, finite
}

// BenchmarkGPUQ8BVectorGEMV compares M serial single-vector Q8B GEMVs with
// the FP32 multi-vector kernel on all real Qwen3-ASR decoder layer
// projections and the vocabulary head. Every variant reads the checkpoint's
// unchanged packed weights/scales; all inputs are nonzero and output buffers
// are observed from shared Metal memory after one submitted command buffer.
func BenchmarkGPUQ8BVectorGEMV(b *testing.B) {
	m := loadOfficialASRQ8B(b, true)
	g, c := m.gpu, &m.cfg
	dev := g.dev
	if err := g.ensureDecodeVec(); err != nil {
		b.Fatal(err)
	}
	qdim := c.heads * c.headDim
	qkvN := qdim + 2*c.kvDim
	gemvParts := c.hidden / gpuRows(g.bits)
	groupRows := gpuRows(g.bits)
	bodyGroups := 2 * c.intermediate / groupRows
	oneLayerBytes := c.hidden*qkvN + 2*qkvN*c.hidden/q4Group +
		qdim*c.hidden + 2*qdim*c.hidden/q4Group +
		2*c.intermediate*c.hidden + 4*c.intermediate*c.hidden/q4Group +
		c.hidden*c.intermediate + 2*c.hidden*c.intermediate/q4Group
	bodyBytes := int64(len(g.layers) * oneLayerBytes)
	headBytes := int64(g.lmScale + 2*g.lmRows*(c.hidden/q4Group))
	logicalBytes := bodyBytes + headBytes

	for _, lanes := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("M=%d", lanes), func(b *testing.B) {
			wi := decodeVectorIndex(lanes)
			vector := struct {
				qkv, o, gateup, down, head *metal.Pipeline
			}{qkv: g.decodeVec[wi][0], o: g.decodeVec[wi][1], gateup: g.decodeVec[wi][2], down: g.decodeVec[wi][3], head: g.decodeVec[wi][4]}

			hidden := batchBenchBuffer(b, dev, 4*lanes*c.hidden)
			ctx := batchBenchBuffer(b, dev, 4*lanes*qdim)
			act := batchBenchBuffer(b, dev, 4*lanes*c.intermediate)
			qkv := batchBenchBuffer(b, dev, 4*lanes*qkvN)
			resid := batchBenchBuffer(b, dev, 4*lanes*c.hidden)
			logits := batchBenchBuffer(b, dev, 4*lanes*g.lmRows)
			inParts := batchBenchBuffer(b, dev, 4*lanes*gemvParts)
			outParts := batchBenchBuffer(b, dev, 4*lanes*bodyGroups)
			rng := rand.New(rand.NewPCG(9001, uint64(lanes)))
			hiddenSums := fillBenchRows(floats(hidden.Bytes()), lanes, c.hidden, rng)
			fillBenchRows(floats(ctx.Bytes()), lanes, qdim, rng)
			fillBenchRows(floats(act.Bytes()), lanes, c.intermediate, rng)
			fillBenchNormParts(floats(inParts.Bytes()), hiddenSums, gemvParts)
			residual := floats(resid.Bytes())
			for i := range residual {
				residual[i] = float32(rng.NormFloat64()) * 0.01
			}

			qArgs := gemvArgs{k: uint32(c.hidden), n: uint32(qkvN), eps: float32(c.eps), parts: uint32(gemvParts)}
			oArgs := gemvArgs{k: uint32(qdim), n: uint32(c.hidden), eps: float32(c.eps)}
			guArgs := gemvArgs{k: uint32(c.hidden), n: uint32(2 * c.intermediate), eps: float32(c.eps), parts: uint32(gemvParts)}
			dArgs := gemvArgs{k: uint32(c.intermediate), n: uint32(c.hidden), eps: float32(c.eps)}
			headArgs := gemvArgs{k: uint32(c.hidden), n: uint32(g.lmRows), eps: float32(c.eps)}
			var enc metal.Encoder
			dispatch := func(p *metal.Pipeline, buf *metal.Buffer, wOff, sOff int,
				x *metal.Buffer, xOff int, y *metal.Buffer, yOff int,
				partsIn *metal.Buffer, inOff int, partsOut *metal.Buffer, outOff int,
				args *gemvArgs, rows int) {
				enc.SetPipeline(p)
				enc.SetBuffer(buf, wOff, 0)
				enc.SetBuffer(buf, sOff, 1)
				enc.SetBuffer(x, xOff, 2)
				enc.SetBuffer(y, yOff, 3)
				enc.SetBuffer(partsIn, inOff, 4)
				enc.SetBytes(unsafe.Pointer(args), 16, 5)
				enc.SetBuffer(partsOut, outOff, 6)
				enc.Dispatch(metal.Size{X: int(args.n) / rows, Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
			}
			serial := func() error {
				dev.Begin(&enc, false)
				for i := range g.layers {
					gl := &g.layers[i]
					for lane := range lanes {
						hOff, ctxOff, actOff := 4*lane*c.hidden, 4*lane*qdim, 4*lane*c.intermediate
						qOff, aOff, hOut := 4*lane*qkvN, actOff, hOff
						partOff := 4 * lane * gemvParts
						outOff := 4 * lane * (c.hidden / groupRows)
						dispatch(g.qkv, gl.buf, gl.qkv, gl.qkvScale, hidden, hOff, qkv, qOff, inParts, partOff, outParts, 0, &qArgs, groupRows)
						dispatch(g.o, gl.buf, gl.o, gl.oScale, ctx, ctxOff, resid, hOut, inParts, 0, outParts, outOff, &oArgs, groupRows)
						dispatch(g.gateup, gl.buf, gl.gu, gl.guScale, hidden, hOff, act, aOff, inParts, partOff, outParts, 0, &guArgs, groupRows)
						dispatch(g.down, gl.buf, gl.d, gl.dScale, act, actOff, resid, hOut, inParts, 0, outParts, outOff, &dArgs, groupRows)
					}
				}
				for lane := range lanes {
					dispatch(g.gemvHead, g.lm, 0, g.lmScale, hidden, 4*lane*c.hidden, logits, 4*lane*g.lmRows,
						inParts, 0, outParts, 0, &headArgs, gpuHeadRows)
				}
				return enc.Wait()
			}
			batch := func() error {
				dev.Begin(&enc, false)
				for i := range g.layers {
					gl := &g.layers[i]
					dispatch(vector.qkv, gl.buf, gl.qkv, gl.qkvScale, hidden, 0, qkv, 0, inParts, 0, outParts, 0, &qArgs, groupRows)
					dispatch(vector.o, gl.buf, gl.o, gl.oScale, ctx, 0, resid, 0, inParts, 0, outParts, 0, &oArgs, groupRows)
					dispatch(vector.gateup, gl.buf, gl.gu, gl.guScale, hidden, 0, act, 0, inParts, 0, outParts, 0, &guArgs, groupRows)
					dispatch(vector.down, gl.buf, gl.d, gl.dScale, act, 0, resid, 0, inParts, 0, outParts, 0, &dArgs, groupRows)
				}
				dispatch(vector.head, g.lm, 0, g.lmScale, hidden, 0, logits, 0, inParts, 0, outParts, 0, &headArgs, gpuHeadRows)
				return enc.Wait()
			}
			consumeOutput := func(b *testing.B) {
				for _, v := range floats(logits.Bytes()) {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						b.Fatal("non-finite vector head output")
					}
				}
			}
			initialResidual := append([]float32(nil), residual...)
			if err := serial(); err != nil {
				b.Fatal(err)
			}
			serialOutputs := [][]float32{
				append([]float32(nil), floats(qkv.Bytes())...),
				append([]float32(nil), floats(act.Bytes())...),
				append([]float32(nil), floats(resid.Bytes())...),
				append([]float32(nil), floats(logits.Bytes())...),
			}
			copy(residual, initialResidual)
			if err := batch(); err != nil {
				b.Fatal(err)
			}
			batchOutputs := [][]float32{floats(qkv.Bytes()), floats(act.Bytes()), floats(resid.Bytes()), floats(logits.Bytes())}
			for i, name := range []string{"QKV", "SwiGLU", "residual", "head"} {
				maxAbs, rel, finite := gpuVectorError(serialOutputs[i], batchOutputs[i])
				if !finite || maxAbs > 1e-4 || rel > 1e-5 {
					b.Fatalf("real-weight %s parity: max_abs %.3g relative %.3g finite=%t", name, maxAbs, rel, finite)
				}
			}

			b.Run("serial-GEMV", func(b *testing.B) {
				if err := serial(); err != nil {
					b.Fatal(err)
				}
				consumeOutput(b)
				b.SetBytes(logicalBytes * int64(lanes))
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					clear(floats(resid.Bytes()))
					if err := serial(); err != nil {
						b.Fatal(err)
					}
					consumeOutput(b)
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(lanes)/1e3, "µs/lane")
			})
			b.Run("vector-GEMV", func(b *testing.B) {
				if err := batch(); err != nil {
					b.Fatal(err)
				}
				consumeOutput(b)
				b.SetBytes(logicalBytes * int64(lanes))
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					clear(floats(resid.Bytes()))
					if err := batch(); err != nil {
						b.Fatal(err)
					}
					consumeOutput(b)
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(lanes)/1e3, "µs/lane")
			})
		})
	}
}

func loadOfficialASRQ8B(tb testing.TB, withHead bool) *Weights {
	tb.Helper()
	path := testmodels.Path(tb, testmodels.Qwen3ASR)
	raw, err := os.ReadFile(path + "/config.json")
	if err != nil {
		tb.Fatal(err)
	}
	var cfg struct {
		Thinker struct {
			Text TextConfig `json:"text_config"`
		} `json:"thinker_config"`
	}
	if err := vibejson.Unmarshal(raw, &cfg); err != nil {
		tb.Fatal(err)
	}
	// The Qwen3-ASR default MRoPE axes coincide for text-only decoder inputs.
	cfg.Thinker.Text.RopeScaling = nil
	opts := LoadOptions{Format: WeightsGPUQ8, Prefix: "thinker.model.", Config: &cfg.Thinker.Text}
	if withHead {
		st, err := safetensors.Open(path)
		if err != nil {
			tb.Fatal(err)
		}
		opts.Head = "thinker.lm_head.weight"
		if !st.Has(opts.Head) {
			opts.Head = "thinker.model.embed_tokens.weight"
		}
		st.Close()
	}
	m, err := Load(path, opts)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(m.Release)
	return m
}
