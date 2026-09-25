// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
	"github.com/GetStream/gophonic/internal/safetensors"
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
			deq[r*k+i] = float32(q) * safetensors.F16ToF32(scales[r*k/q4Group+i/32])
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
	for _, name := range []string{"mm_o_16", "mm_o_w", "mm_o_q8_w", "mm_o_q4_w"} {
		p, err := dev.Pipeline(lib, name)
		if err != nil {
			b.Fatal(err)
		}
		tile, threads := 32, 256
		switch {
		case name == "mm_o_16":
			tile, threads = 16, 128
		case strings.HasSuffix(name, "_w"):
			threads = 128
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
						e.Dispatch(metal.Size{X: n / mmColumns, Y: (rows + tile - 1) / tile, Z: 1}, metal.Size{X: threads, Y: 1, Z: 1})
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

// BenchmarkGPUGemvShapes measures serial GEMV dispatches of the Qwen3-1.7B
// projection shapes over 28 weight copies each, as one decode step issues
// them.
func BenchmarkGPUGemvShapes(b *testing.B) {
	dev, err := metal.Open()
	if err != nil {
		b.Skip(err)
	}
	c := &modelConfig{hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, kvDim: 1024, intermediate: 6144}
	lib, err := dev.Compile(gpuSourceFor(c))
	if err != nil {
		b.Fatal(err)
	}
	const mats = 28
	for _, s := range []struct {
		name string
		k, n int
	}{{"qkv", 2048, 4096}, {"o", 2048, 2048}, {"gateup", 2048, 12288}, {"down", 6144, 2048}} {
		for _, kernel := range []string{"gemv_o", "gemv_o_q8"} {
			b.Run(s.name+"/"+kernel, func(b *testing.B) {
				p, err := dev.Pipeline(lib, kernel)
				if err != nil {
					b.Fatal(err)
				}
				wBytes, sBytes := s.n*s.k, 2*s.n*s.k/q4Group
				w, _ := dev.Buffer(wBytes * mats)
				sc, _ := dev.Buffer(sBytes * mats)
				x, _ := dev.Buffer(4 * s.k)
				y, _ := dev.Buffer(4 * s.n)
				parts, _ := dev.Buffer(4 * s.n)
				args := gemvArgs{k: uint32(s.k), n: uint32(s.n)}
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
					for m := range mats {
						e.SetBuffer(w, m*wBytes, 0)
						e.SetBuffer(sc, m*sBytes, 1)
						e.Dispatch(metal.Size{X: s.n / gpuRows(9), Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
					}
					if err := e.Wait(); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N)/mats, "µs/dispatch")
			})
		}
	}
}
