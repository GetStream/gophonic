// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"testing"
	"time"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
)

// TestGPUGemvQ4 checks the 4-bit GEMV against a dequantized CPU product.
func TestGPUGemvQ4(t *testing.T) {
	dev, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	lib, err := dev.Compile(gpuSource)
	if err != nil {
		t.Fatal(err)
	}
	p, err := dev.Pipeline(lib, "gemv_o_q4")
	if err != nil {
		t.Fatal(err)
	}
	const n, k = 64, 4096
	rng := rand.New(rand.NewPCG(1, 2))
	w, _ := dev.Buffer(n * k / 2)
	sc, _ := dev.Buffer(2 * n * k / q4Group)
	x, _ := dev.Buffer(4 * k)
	y, _ := dev.Buffer(4 * n)
	parts, _ := dev.Buffer(4 * n)
	scales := unsafe.Slice((*uint16)(unsafe.Pointer(&sc.Bytes()[0])), n*k/q4Group)
	row := make([]float32, k)
	deq := make([]float32, n*k)
	for r := range n {
		for i := range row {
			row[i] = float32(rng.NormFloat64()) * 0.02
		}
		quantizeRowQ4(row, w.Bytes()[r*k/2:(r+1)*k/2], scales[r*k/q4Group:(r+1)*k/q4Group], 1)
		for i := range k {
			b := w.Bytes()[r*k/2+i/32*16+i%16]
			q := int(b&15) - 8
			if i%32 >= 16 {
				q = int(b>>4) - 8
			}
			deq[r*k+i] = float32(q) * f16Bits(scales[r*k/q4Group+i/32])
		}
	}
	xs := floats(x.Bytes())
	for i := range xs {
		xs[i] = float32(rng.NormFloat64())
	}
	args := gemvArgs{k: k, n: n}
	var e metal.Encoder
	dev.Begin(&e, false)
	e.SetPipeline(p)
	e.SetBuffer(w, 0, 0)
	e.SetBuffer(sc, 0, 1)
	e.SetBuffer(x, 0, 2)
	e.SetBuffer(y, 0, 3)
	e.SetBuffer(parts, 0, 4)
	e.SetBytes(unsafe.Pointer(&args), 16, 5)
	e.SetBuffer(parts, 0, 6)
	e.Dispatch(metal.Size{X: n / gpuRows(4), Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
	if err := e.Wait(); err != nil {
		t.Fatal(err)
	}
	ys := floats(y.Bytes())
	var errSum, refSum float64
	for r := range n {
		var want float64
		for i := range k {
			want += float64(deq[r*k+i]) * float64(xs[i])
		}
		errSum += (float64(ys[r]) - want) * (float64(ys[r]) - want)
		refSum += want * want
		if r < 3 {
			t.Logf("row %d: gpu %.5f cpu %.5f", r, ys[r], want)
		}
	}
	if rel := math.Sqrt(errSum / refSum); rel > 1e-5 {
		t.Fatalf("relative error %.3g", rel)
	}
	// Quantization error itself.
	var qe, qr float64
	rng2 := rand.New(rand.NewPCG(1, 2))
	for r := range n {
		for i := range k {
			v := float64(float32(rng2.NormFloat64()) * 0.02)
			d := float64(deq[r*k+i]) - v
			qe += d * d
			qr += v * v
		}
	}
	t.Logf("weight SNR %.1f dB", 10*math.Log10(qr/qe))
}

// BenchmarkGPUGemvStream measures weight streaming bandwidth of the O-shape
// GEMV kernels over a buffer larger than the system cache.
func BenchmarkGPUGemvStream(b *testing.B) {
	dev, err := metal.Open()
	if err != nil {
		b.Skip(err)
	}
	lib, err := dev.Compile(gpuSource)
	if err != nil {
		b.Fatal(err)
	}
	const k, n, mats = 4096, 4096, 96
	for _, c := range []struct {
		bits int
		one  bool
	}{{8, false}, {4, false}, {8, true}, {4, true}} {
		bits := c.bits
		b.Run(fmt.Sprintf("bits=%d/one-dispatch=%v", bits, c.one), func(b *testing.B) {
			name := "gemv_o"
			wBytes, sBytes := n*k, 4*n
			if bits == 4 {
				name, wBytes, sBytes = "gemv_o_q4", n*k/2, 2*n*k/q4Group
			}
			p, err := dev.Pipeline(lib, name)
			if err != nil {
				b.Fatal(err)
			}
			w, _ := dev.Buffer(wBytes * mats)
			sc, _ := dev.Buffer(sBytes * mats)
			x, _ := dev.Buffer(4 * k)
			y, _ := dev.Buffer(4 * n * mats)
			parts, _ := dev.Buffer(4 * n * mats)
			for i, wb := 0, w.Bytes(); i < len(wb); i++ {
				wb[i] = byte(i * 7 % 13)
			}
			for i, sb := 0, sc.Bytes(); i < len(sb); i += 2 {
				sb[i+1] = 0x3c // FP16 1.0, or a small FP32 value
			}
			args := gemvArgs{k: k, n: n}
			rows := gpuRows(bits)
			var e metal.Encoder
			b.SetBytes(int64((wBytes + sBytes) * mats))
			for b.Loop() {
				dev.Begin(&e, false)
				e.SetPipeline(p)
				e.SetBuffer(x, 0, 2)
				e.SetBuffer(y, 0, 3)
				e.SetBuffer(parts, 0, 4)
				e.SetBytes(unsafe.Pointer(&args), 16, 5)
				e.SetBuffer(parts, 0, 6)
				if c.one {
					e.SetBuffer(w, 0, 0)
					e.SetBuffer(sc, 0, 1)
					e.Dispatch(metal.Size{X: n * mats / rows, Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
				}
				for m := range mats {
					if c.one {
						break
					}
					e.SetBuffer(w, m*wBytes, 0)
					e.SetBuffer(sc, m*sBytes, 1)
					e.Dispatch(metal.Size{X: n / rows, Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
				}
				if err := e.Wait(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestOfficialGPUBatchMatchesTokens compares the batched GPU forward pass with
// the token-by-token one on a real checkpoint.
func TestOfficialGPUBatchMatchesTokens(t *testing.T) {
	path := os.Getenv("GOPHONIC_QWEN3_MODEL")
	if path == "" {
		t.Skip("set GOPHONIC_QWEN3_MODEL")
	}
	for _, format := range []string{WeightsGPU, WeightsGPUQ4} {
		if only := os.Getenv("GOPHONIC_QWEN_WEIGHTS"); only != "" && only != format {
			continue
		}
		m, err := Open(path, Options{Weights: format, CacheEntries: -1})
		if err != nil {
			t.Fatal(err)
		}
		ids, err := m.tokens.EncodeInto("The quick brown fox jumps over the lazy dog while the band plays a slow song about rivers and mountains far away.", make([]int, 0, 256), &m.tokenWS)
		if err != nil {
			t.Fatal(err)
		}
		batch := make([]float32, hiddenSize)
		tokens := make([]float32, hiddenSize)
		if err := m.eval.HiddenLastInto(ids, batch, m.ws); err != nil {
			t.Fatal(err)
		}
		gpuTokenByToken = true
		err = m.eval.HiddenLastInto(ids, tokens, m.ws)
		gpuTokenByToken = false
		if err != nil {
			t.Fatal(err)
		}
		cos, maxAbs := vectorParity(batch, tokens)
		t.Logf("%s: %d tokens, batched vs token-by-token cosine %.7f (max_abs %.3g)", format, len(ids), cos, maxAbs)
		if cos < 0.99999 {
			t.Errorf("%s: batched path diverges: cosine %.7f", format, cos)
		}
		// Packing several sequences into one pass matches evaluating each.
		seqs := [][]int{ids[3:9], ids, ids[:1]}
		packed := [][]float32{make([]float32, hiddenSize), make([]float32, hiddenSize), make([]float32, hiddenSize)}
		if err := m.eval.HiddenLastBatchInto(seqs, packed, m.ws); err != nil {
			t.Fatal(err)
		}
		for s, seq := range seqs {
			alone := make([]float32, hiddenSize)
			if err := m.eval.HiddenLastInto(seq, alone, m.ws); err != nil {
				t.Fatal(err)
			}
			cos, _ := vectorParity(packed[s], alone)
			if cos < 0.99999 {
				t.Errorf("%s: packed sequence %d diverges: cosine %.7f", format, s, cos)
			}
		}
		m.Close()
	}
}

// BenchmarkGPUMatMul measures the batched projection kernels on a
// 4096×4096 weight for several token counts.
func BenchmarkGPUMatMul(b *testing.B) {
	dev, err := metal.Open()
	if err != nil {
		b.Skip(err)
	}
	lib, err := dev.Compile(gpuSource)
	if err != nil {
		b.Fatal(err)
	}
	const k, n, reps = 4096, 4096, 16
	for _, name := range []string{"mm_o_16", "mm_o", "mm_o_q4"} {
		p, err := dev.Pipeline(lib, name)
		if err != nil {
			b.Fatal(err)
		}
		tile := 32
		if name == "mm_o_16" {
			tile = 16
		}
		for _, rows := range []int{16, 32, 64, 128, 256} {
			b.Run(fmt.Sprintf("%s/M=%d", name, rows), func(b *testing.B) {
				w, _ := dev.Buffer(n * k)
				sc, _ := dev.Buffer(4 * n * k / 32)
				x, _ := dev.Buffer(4 * rows * k)
				y, _ := dev.Buffer(4 * rows * n)
				parts, _ := dev.Buffer(4 * rows * n)
				args := mmArgs{k: k, n: n, m: uint32(rows), partsOut: n / mmColumns, splitK: k, splits: 1}
				var e metal.Encoder
				for b.Loop() {
					dev.Begin(&e, false)
					e.SetPipeline(p)
					e.SetBuffer(w, 0, 0)
					e.SetBuffer(sc, 0, 1)
					e.SetBuffer(x, 0, 2)
					e.SetBuffer(y, 0, 3)
					e.SetBuffer(parts, 0, 4)
					e.SetBytes(unsafe.Pointer(&args), int(unsafe.Sizeof(args)), 5)
					e.SetBuffer(parts, 0, 6)
					e.SetBuffer(parts, 0, 7)
					for range reps {
						e.Dispatch(metal.Size{X: n / mmColumns, Y: (rows + tile - 1) / tile, Z: 1}, metal.Size{X: 8 * tile, Y: 1, Z: 1})
					}
					if err := e.Wait(); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(2*reps*rows*n*k)/(float64(b.Elapsed().Nanoseconds())/float64(b.N))/1e3, "TFLOPS")
			})
		}
	}
}

func TestOfficialGPUScaling(t *testing.T) {
	path := os.Getenv("GOPHONIC_QWEN3_MODEL")
	if path == "" || os.Getenv("GOPHONIC_QWEN3_GPU_SCALING") == "" {
		t.Skip("set GOPHONIC_QWEN3_MODEL and GOPHONIC_QWEN3_GPU_SCALING")
	}
	m, err := Open(path, Options{Weights: os.Getenv("GOPHONIC_QWEN_WEIGHTS"), CacheEntries: -1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	dst := make([]float32, hiddenSize)
	for _, n := range []int{1, 12, 16, 17, 32, 64, 128, 192} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = 1000 + i*37
		}
		_ = m.eval.HiddenLastInto(ids, dst, m.ws)
		best := time.Hour
		for range 3 {
			start := time.Now()
			if err := m.eval.HiddenLastInto(ids, dst, m.ws); err != nil {
				t.Fatal(err)
			}
			best = min(best, time.Since(start))
		}
		t.Logf("%4d tokens: %v", n, best.Round(100*time.Microsecond))
	}
}

// TestOfficialGPUFlashAttention compares the tiled attention kernel with the
// per-key loop on packed sequences, a long input, and a shared prefix.
func TestOfficialGPUFlashAttention(t *testing.T) {
	path := os.Getenv("GOPHONIC_QWEN3_MODEL")
	if path == "" {
		t.Skip("set GOPHONIC_QWEN3_MODEL")
	}
	m, err := Open(path, Options{Weights: WeightsGPU, CacheEntries: -1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	long := make([]int, 150)
	for i := range long {
		long[i] = 1000 + (i*7919)%50000
	}
	seqs := [][]int{long[:5], long, long[20:31]}
	run := func(scalar bool) [][]float32 {
		gpuScalarAttention = scalar
		defer func() { gpuScalarAttention = false }()
		out := [][]float32{make([]float32, hiddenSize), make([]float32, hiddenSize), make([]float32, hiddenSize)}
		if err := m.eval.HiddenLastBatchInto(seqs, out, m.ws); err != nil {
			t.Fatal(err)
		}
		return out
	}
	flash, scalar := run(false), run(true)
	for s := range seqs {
		cos, maxAbs := vectorParity(flash[s], scalar[s])
		t.Logf("sequence %d (%d tokens): cosine %.7f max_abs %.3g", s, len(seqs[s]), cos, maxAbs)
		if cos < 0.99999 {
			t.Errorf("sequence %d: flash attention diverges", s)
		}
	}
	q, err := m.Question("Which support team should handle this customer message?",
		[]string{"payments", "cancellations", "technical support", "shipping", "account login"})
	if err != nil {
		t.Fatal(err)
	}
	texts := benchmarkTexts(4, 14)
	probs := func(scalar bool) [][]float32 {
		gpuScalarAttention = scalar
		defer func() { gpuScalarAttention = false }()
		rows := make([][]float32, len(texts))
		for i := range rows {
			rows[i] = make([]float32, 5)
		}
		if err := q.ChooseBatch(context.Background(), texts, rows); err != nil {
			t.Fatal(err)
		}
		return rows
	}
	pf, ps := probs(false), probs(true)
	for i := range pf {
		for j := range pf[i] {
			if d := math.Abs(float64(pf[i][j] - ps[i][j])); d > 5e-3 { // reassociation, amplified by the letter head
				t.Errorf("input %d option %d: flash %.6f scalar %.6f", i, j, pf[i][j], ps[i][j])
			}
		}
	}
}
