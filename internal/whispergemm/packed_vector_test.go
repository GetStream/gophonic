// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"io"
	"math"
	"runtime"
	"runtime/pprof"
	"sync"
	"testing"
)

func TestHalfConversionExhaustive(t *testing.T) {
	for h := 0; h < 1<<16; h++ {
		v := float32FromHalf(uint16(h))
		if v != v {
			if _, ok := halfFromFloat32(v); ok {
				t.Fatalf("NaN %04x accepted", h)
			}
			continue
		}
		got, ok := halfFromFloat32(v)
		if !ok || got != uint16(h) {
			t.Fatalf("half %04x -> %x -> %04x ok=%v", h, math.Float32bits(v), got, ok)
		}
	}
	for _, v := range []float32{1 + 1.0/2048, 65520, 1e-8, math.SmallestNonzeroFloat32, 0.1} {
		if _, ok := halfFromFloat32(v); ok {
			t.Fatalf("%g accepted as exact FP16", v)
		}
	}
}

func vectorReference(weights []float32, stride, rows, k int, x []float32) []float32 {
	out := make([]float32, rows)
	for r := range out {
		var sum float32
		for c := 0; c < k; c++ {
			sum = float32(math.FMA(float64(x[c]), float64(weights[r*stride+c]), float64(sum)))
		}
		out[r] = sum
	}
	return out
}

func vectorCase(rows, k int, half bool, seed uint32) (weights, x []float32, stride int) {
	stride = k + 3
	weights = make([]float32, rows*stride)
	x = make([]float32, k+5)
	for i := range weights {
		v := nextValue(&seed)
		if half {
			h := uint16(nextValue(&seed)*20000) & 0x7bff // finite, both signs via bit 15 below
			if seed&1 != 0 {
				h |= 0x8000
			}
			v = float32FromHalf(h &^ 0x4000) // magnitudes below 2
		}
		weights[i] = v
	}
	for i := range x {
		x[i] = nextValue(&seed)
	}
	return weights, x, stride
}

func TestPackedVectorSequentialFMA(t *testing.T) {
	for _, half := range []bool{false, true} {
		for _, shape := range [][2]int{
			{1, 1}, {3, 5}, {64, 4}, {65, 7}, {127, 64}, {384, 384}, {511, 3},
			{512, 384}, {513, 385}, {1536, 384}, {384, 1536}, {1500, 64}, {64, 1500},
			{4097, 33},
		} {
			rows, k := shape[0], shape[1]
			weights, x, stride := vectorCase(rows, k, half, uint32(rows*7919+k))
			p, err := NewPackedVector(weights, stride, rows, k)
			if err != nil {
				t.Fatal(err)
			}
			if p.Half() != half {
				t.Fatalf("rows=%d k=%d half=%v stored half=%v", rows, k, half, p.Half())
			}
			want := vectorReference(weights, stride, rows, k, x)
			got := make([]float32, rows+9)
			for i := range got {
				got[i] = -317
			}
			if err := p.Mul(got, x); err != nil {
				t.Fatal(err)
			}
			for i := range got {
				w := float32(-317)
				if i < rows {
					w = want[i]
				}
				if math.Float32bits(got[i]) != math.Float32bits(w) {
					t.Fatalf("half=%v rows=%d k=%d index %d: %x != %x", half, rows, k, i,
						math.Float32bits(got[i]), math.Float32bits(w))
				}
			}
			// The generic path shares the layout and must agree exactly.
			generic := make([]float32, rows)
			p.mulGeneric(generic, x)
			for i := range generic {
				if math.Float32bits(generic[i]) != math.Float32bits(want[i]) {
					t.Fatalf("generic half=%v rows=%d k=%d index %d", half, rows, k, i)
				}
			}
			split := make([]float32, rows)
			for c := 0; c < p.Chunks(); c++ {
				if err := p.MulChunks(split, x, c, c+1); err != nil {
					t.Fatal(err)
				}
			}
			for i := range split {
				if math.Float32bits(split[i]) != math.Float32bits(want[i]) {
					t.Fatalf("chunked half=%v rows=%d k=%d index %d", half, rows, k, i)
				}
			}
			if allocs := testing.AllocsPerRun(3, func() { _ = p.Mul(got, x) }); allocs != 0 {
				t.Fatalf("allocations: %g", allocs)
			}
		}
	}
}

func TestPackedVectorRepack(t *testing.T) {
	weights, x, stride := vectorCase(700, 64, false, 3)
	p, err := NewPackedVectorFP32(700, 64)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Repack(weights, stride); err != nil {
		t.Fatal(err)
	}
	want := vectorReference(weights, stride, 700, 64, x)
	got := make([]float32, 700)
	if err := p.Mul(got, x); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("index %d", i)
		}
	}
}

func TestPackedVectorRepackShape(t *testing.T) {
	p, err := NewPackedVectorFP32(1500, 64)
	if err != nil {
		t.Fatal(err)
	}
	for _, shape := range [][2]int{{1500, 64}, {7, 64}, {64, 1500}, {3, 1}, {1500, 64}} {
		rows, k := shape[0], shape[1]
		weights, x, stride := vectorCase(rows, k, false, uint32(rows+k))
		if err := p.RepackShape(weights, stride, rows, k); err != nil {
			t.Fatal(err)
		}
		want := vectorReference(weights, stride, rows, k, x)
		got := make([]float32, rows)
		if err := p.Mul(got, x); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("rows=%d k=%d index %d", rows, k, i)
			}
		}
	}
	if err := p.RepackShape(make([]float32, 1600*65), 65, 1600, 65); err == nil {
		t.Fatal("accepted a shape larger than the allocation")
	}
}

func TestPackedVectorRepackColumns(t *testing.T) {
	p, _ := NewPackedVectorFP32(1500, 64)
	for _, shape := range [][2]int{{1500, 64}, {64, 1500}, {70, 9}} {
		rows, k := shape[0], shape[1]
		stride := rows + 5
		src := make([]float32, k*stride)
		state := uint32(rows*31 + k)
		for i := range src {
			src[i] = nextValue(&state)
		}
		bias := make([]float32, rows)
		for i := range bias {
			bias[i] = nextValue(&state)
		}
		for _, variant := range []struct {
			scale float32
			bias  []float32
		}{{1, nil}, {0.35355338, nil}, {1, bias}} {
			if err := p.RepackColumns(src, stride, rows, k, variant.scale, variant.bias); err != nil {
				t.Fatal(err)
			}
			weights := make([]float32, rows*k)
			for r := 0; r < rows; r++ {
				for c := 0; c < k; c++ {
					v := src[c*stride+r]
					if variant.scale != 1 {
						v *= variant.scale
					}
					if variant.bias != nil {
						v += variant.bias[r]
					}
					weights[r*k+c] = v
				}
			}
			x := make([]float32, k)
			for i := range x {
				x[i] = nextValue(&state)
			}
			want := vectorReference(weights, k, rows, k, x)
			got := make([]float32, rows)
			if err := p.Mul(got, x); err != nil {
				t.Fatal(err)
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("rows=%d k=%d scale=%v bias=%v index %d", rows, k, variant.scale, variant.bias != nil, i)
				}
			}
		}
	}
}

func TestPackedVectorSignalStorm(t *testing.T) {
	if !PackedVectorAccelerated() || testing.Short() {
		t.Skip("SME signal stress")
	}
	if err := pprof.StartCPUProfile(io.Discard); err == nil {
		defer pprof.StopCPUProfile()
	}
	for _, half := range []bool{false, true} {
		weights, x, stride := vectorCase(5000, 384, half, 11)
		p, _ := NewPackedVector(weights, stride, 5000, 384)
		want := vectorReference(weights, stride, 5000, 384, x)
		before := smeRetries.Load()
		var wg sync.WaitGroup
		var bad sync.Once
		failed := false
		for g := 0; g < 2*runtime.GOMAXPROCS(0)+3; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				got := make([]float32, 5000)
				for it := 0; it < 200; it++ {
					_ = p.Mul(got, x)
					for i := range got {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							bad.Do(func() { failed = true })
							return
						}
					}
					if it%16 == 0 {
						runtime.GC()
					}
				}
			}()
		}
		wg.Wait()
		if failed {
			t.Fatalf("half=%v mismatch under signals", half)
		}
		t.Logf("half=%v chunks/stores redone after signals: %d", half, smeRetries.Load()-before)
	}
}

func BenchmarkVocabularyProjection(b *testing.B) {
	const rows, k = 51864, 384
	for _, half := range []bool{false, true} {
		weights, x, stride := vectorCase(rows, k, half, 5)
		name := "fp32"
		if half {
			name = "fp16"
		}
		b.Run("packed-"+name, func(b *testing.B) {
			p, _ := NewPackedVector(weights, stride, rows, k)
			dst := make([]float32, rows)
			for b.Loop() {
				_ = p.Mul(dst, x)
			}
		})
		if !half {
			b.Run("MulVector-fp32", func(b *testing.B) {
				dst := make([]float32, rows)
				for b.Loop() {
					_ = MulVector(dst, weights, stride, x[:k], rows)
				}
			})
		}
	}
}
