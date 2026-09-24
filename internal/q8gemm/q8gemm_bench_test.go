// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import (
	"math/rand"
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

func BenchmarkQ8GEMM16x64(b *testing.B) {
	for _, shape := range []struct{ m, k, n int }{
		{12, 4096, 4096},  // Q/O projections
		{12, 4096, 1024},  // K/V projections
		{12, 4096, 12288}, // Gate/Up projections
		{12, 12288, 4096}, // Down projection
	} {
		b.Run(shapeName(shape.m, shape.k, shape.n), func(b *testing.B) {
			benchQ8GEMMShape(b, shape.m, shape.k, shape.n)
		})
	}
}

func benchQ8GEMMShape(b *testing.B, m, k, n int) {
	b.Helper()
	// Three packed matrices rotate through more than the CPU's private cache.
	rotations := 3
	if k*n <= 8<<20 {
		rotations = 12 // keep even K/V's complete backing set larger than cache
	}
	weights := make([]*Weights, rotations)
	for i := range weights {
		weights[i] = benchmarkWeights(b, k, n, int64(i+1))
	}
	x := make([]float32, m*k)
	rng := rand.New(rand.NewSource(99))
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	dst := make([]float32, m*n)
	ws, err := NewWorkspace(k)
	if err != nil {
		b.Fatal(err)
	}
	b.Run("SME_or_scalar", func(b *testing.B) {
		b.ReportAllocs()
		if !usingSME() && os.Getenv("Q8GEMM_REQUIRE_SME") == "1" {
			b.Fatal("Q8GEMM_REQUIRE_SME is set but SME 512-bit dispatch is inactive")
		}
		if usingSME() {
			b.Log("dispatch=SME16x64, SVL=64")
		} else {
			b.Log("dispatch=scalar (SME unavailable)")
		}
		b.SetBytes(int64(k * n))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := MulInto(dst, x, m, weights[i%rotations], ws); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("scalar_packed", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(k * n))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			w := weights[i%rotations]
			if err := ws.Pack(x, m, k); err != nil {
				b.Fatal(err)
			}
			scalarMulPanels(dst, w.n, ws, w, 0, w.panels)
		}
	})
	b.Run("expanded_FP32_SME", func(b *testing.B) {
		// This setup deliberately materializes and packs an FP32 copy. It is a
		// strong throughput reference, not a memory-efficient runtime design.
		expandedRotations := 1
		if k*n <= 8<<20 {
			expandedRotations = 4 // the smaller expanded matrices otherwise stay cache-hot
		}
		packed := make([]*whispergemm.PackedB, expandedRotations)
		for rotation := range packed {
			w := weights[rotation%rotations]
			expanded := make([]float32, k*n)
			for row := range n {
				for kk := range k {
					expanded[row*k+kk] = w.at(row, kk) * w.scales[row]
				}
			}
			packed[rotation], err = whispergemm.NewPackedB(k, n)
			if err != nil {
				b.Fatal(err)
			}
			if err = packed[rotation].Pack(expanded, k, true); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportAllocs()
		b.SetBytes(int64(k * n * 4))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := packed[i%len(packed)].Mul(dst, n, x, k, m); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func benchmarkWeights(b *testing.B, k, n int, seed int64) *Weights {
	b.Helper()
	q := make([]int8, k*n)
	rng := rand.New(rand.NewSource(seed))
	for i := range q {
		q[i] = int8(rng.Intn(255) - 127)
	}
	scales := make([]float32, n)
	for i := range scales {
		scales[i] = 1.0 / 127
	}
	w, err := NewWeights(k, n)
	if err != nil {
		b.Fatal(err)
	}
	if err = w.Pack(q, scales); err != nil {
		b.Fatal(err)
	}
	return w
}
