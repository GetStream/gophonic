// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3asr

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
)

// BenchmarkGPUGEMM measures the encoder's matrix-product kernel on the
// shapes of Qwen3-ASR-1.7B: the layer projections at 143 rows (11 s of
// audio) and a convolution-sized product.
func BenchmarkGPUGEMM(b *testing.B) {
	dev, err := metal.Open()
	if err != nil {
		b.Skip(err)
	}
	lib, err := dev.Compile(encoderSource)
	if err != nil {
		b.Fatal(err)
	}
	p, err := dev.Pipeline(lib, "gemm_bias")
	if err != nil {
		b.Fatal(err)
	}
	finish, err := dev.Pipeline(lib, "gemm_finish_bias")
	if err != nil {
		b.Fatal(err)
	}
	for _, s := range []struct{ m, n, k int }{
		{143, 3072, 1024}, {143, 1024, 1024}, {143, 4096, 1024}, {143, 1024, 4096}, {8800, 480, 4320},
		{128, 4096, 4096}, {256, 4096, 4096}, {158, 4096, 2048}, {158, 12288, 2048},
	} {
		b.Run(fmt.Sprintf("M=%d/N=%d/K=%d/splits=%d", s.m, s.n, s.k, gemmSplits(s.m, s.n, s.k)), func(b *testing.B) {
			x, _ := dev.Buffer(4 * s.m * s.k)
			w, _ := dev.Buffer(2 * s.n * s.k)
			scale, _ := dev.Buffer(4 * s.n)
			bias, _ := dev.Buffer(4 * s.n)
			y, _ := dev.Buffer(4 * s.m * s.n)
			table, _ := dev.Buffer(16)
			scratch, _ := dev.Buffer(4 * max(1, gemmScratch(s.m, s.n, s.k)))
			splits := gemmSplits(s.m, s.n, s.k)
			padN := (s.n + gemmTile - 1) / gemmTile * gemmTile
			args := gemmArgs{m: uint32(s.m), n: uint32(s.n), k: uint32(s.k), splitK: uint32(s.k / splits),
				splits: uint32(splits), padN: uint32(padN)}
			var conv convArgs
			const reps = 8
			var e metal.Encoder
			for b.Loop() {
				dev.Begin(&e, false)
				e.SetBuffer(x, 0, 0)
				e.SetBuffer(w, 0, 1)
				e.SetBuffer(scale, 0, 2)
				e.SetBuffer(bias, 0, 3)
				e.SetBuffer(y, 0, 4)
				e.SetBytes(unsafe.Pointer(&args), int(unsafe.Sizeof(args)), 5)
				e.SetBytes(unsafe.Pointer(&conv), int(unsafe.Sizeof(conv)), 6)
				e.SetBuffer(table, 0, 7)
				e.SetBuffer(scratch, 0, 8)
				for range reps {
					e.SetPipeline(p)
					e.Dispatch(metal.Size{X: padN / gemmTile, Y: (s.m + gemmTile - 1) / gemmTile, Z: splits}, metal.Size{X: gemmThreads, Y: 1, Z: 1})
					if splits > 1 {
						e.SetPipeline(finish)
						e.Dispatch(metal.Size{X: padN / 64, Y: s.m, Z: 1}, metal.Size{X: 64, Y: 1, Z: 1})
					}
				}
				if err := e.Wait(); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(2*reps*s.m*s.n*s.k)/(float64(b.Elapsed().Nanoseconds())/float64(b.N))/1e3, "TFLOPS")
		})
	}
}
