// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"fmt"
	"testing"
)

var benchmarkSink float32

var benchmarkShapes = []struct {
	name    string
	m, k, n int
}{
	{"encoder_projection", 1500, 384, 384},
	{"encoder_expand", 1500, 384, 1536},
	{"encoder_contract", 1500, 1536, 384},
	{"attention_scores", 1500, 64, 1500},
	{"attention_values", 1500, 1500, 64},
	{"decoder_projection", 1, 384, 384},
	{"decoder_expand", 1, 384, 1536},
	{"decoder_contract", 1, 1536, 384},
	{"decoder_vocabulary", 1, 384, 51864},
	{"tails", 7, 65, 19},
}

func benchmarkData(tb testing.TB, m, k, n int) (a, w, dst []float32, packed *PackedB) {
	tb.Helper()
	a, w, dst = make([]float32, m*k), make([]float32, n*k), make([]float32, m*n)
	state := uint32(0xc001cafe)
	for i := range a {
		a[i] = nextValue(&state)
	}
	for i := range w {
		w[i] = nextValue(&state)
	}
	var err error
	packed, err = NewPackedB(k, n)
	if err != nil {
		tb.Fatal(err)
	}
	if err := packed.Pack(w, k, true); err != nil {
		tb.Fatal(err)
	}
	if err := packed.Mul(dst, n, a, k, m); err != nil {
		tb.Fatal(err)
	}
	return
}

func BenchmarkMul(b *testing.B) {
	for _, s := range benchmarkShapes {
		b.Run(s.name, func(b *testing.B) {
			a, _, dst, packed := benchmarkData(b, s.m, s.k, s.n)
			b.ReportAllocs()
			for b.Loop() {
				if err := packed.Mul(dst, s.n, a, s.k, s.m); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(2*s.m*s.k*s.n)/b.Elapsed().Seconds()*float64(b.N)/1e9, "GFLOP/s")
			benchmarkSink = dst[len(dst)-1]
		})
	}
}

func BenchmarkExecutor(b *testing.B) {
	for _, workers := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("workers_%d", workers), func(b *testing.B) {
			e, err := NewExecutor(workers)
			if err != nil {
				b.Fatal(err)
			}
			defer e.Close()
			for _, s := range benchmarkShapes {
				b.Run(s.name, func(b *testing.B) {
					a, _, dst, packed := benchmarkData(b, s.m, s.k, s.n)
					if err := e.Mul(packed, dst, s.n, a, s.k, s.m); err != nil {
						b.Fatal(err)
					}
					b.ReportAllocs()
					for b.Loop() {
						if err := e.Mul(packed, dst, s.n, a, s.k, s.m); err != nil {
							b.Fatal(err)
						}
					}
					b.ReportMetric(float64(2*s.m*s.k*s.n)/b.Elapsed().Seconds()*float64(b.N)/1e9, "GFLOP/s")
					benchmarkSink = dst[len(dst)-1]
				})
			}
		})
	}
}

func BenchmarkScalar(b *testing.B) {
	for _, s := range benchmarkShapes {
		b.Run(s.name, func(b *testing.B) {
			a, _, dst, packed := benchmarkData(b, s.m, s.k, s.n)
			b.ReportAllocs()
			for b.Loop() {
				mulPackedScalar(dst, s.n, a, s.k, packed.data, s.m, s.k, s.n)
			}
			benchmarkSink = dst[len(dst)-1]
		})
	}
}

func BenchmarkPack(b *testing.B) {
	for _, s := range benchmarkShapes {
		if s.m != 1 {
			continue
		}
		b.Run(s.name, func(b *testing.B) {
			_, weights, _, packed := benchmarkData(b, s.m, s.k, s.n)
			b.SetBytes(int64(4 * s.k * s.n))
			b.ReportAllocs()
			for b.Loop() {
				if err := packed.Pack(weights, s.k, true); err != nil {
					b.Fatal(err)
				}
			}
			benchmarkSink = packed.data[len(packed.data)-1]
		})
	}
}
