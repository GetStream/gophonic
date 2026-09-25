// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

func TestCompactRowMatchesGeneralTile(t *testing.T) {
	if !Available() {
		t.Skip("compact row requires SME")
	}
	for _, k := range []int{16, 32, 96, 256, 1024, 2048, 6144} {
		normal, _ := NewWorkspace(k)
		compact, _ := NewWorkspace(k)
		rng := rand.New(rand.NewSource(int64(k)))
		x := make([]float32, k)
		for trial := range 6 {
			for i := range x {
				x[i] = math.Float32frombits(rng.Uint32())
			}
			if trial < 4 {
				for i := range x {
					x[i] = float32(rng.NormFloat64()) * float32(math.Ldexp(1, trial*30-45))
				}
			}
			x[0] = math.Float32frombits(0x80000000)
			if err := normal.Pack(x, 1, k); err != nil {
				t.Fatal(err)
			}
			if err := compact.PrepareRowF16(k); err != nil {
				t.Fatal(err)
			}
			compact.SetRowScale(0, MaxAbs(x))
			for i := range compact.activation {
				compact.activation[i] = 0x5a5a
			}
			// Exercise arbitrary legal subranges, including scalar tails.
			for first := 0; first < k; {
				last := min(k, first+2*(1+rng.Intn(19)))
				if err := compact.PackRange(x, k, first, last); err != nil {
					t.Fatal(err)
				}
				first = last
			}
			for i := range x {
				a, b := normal.row[i], compact.row[i]
				// Scalar half conversion canonicalizes NaNs while hardware preserves
				// payload bits. Compare finite/Inf/zero encodings exactly.
				if a != b && !(a&0x7c00 == 0x7c00 && a&0x3ff != 0 && b&0x7c00 == 0x7c00 && b&0x3ff != 0) {
					t.Fatalf("K=%d trial=%d at %d: %04x != %04x", k, trial, i, a, b)
				}
			}
			for _, v := range compact.activation {
				if v != 0x5a5a {
					t.Fatal("compact packing touched the unused tile")
				}
			}
		}
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		bf := make([]uint16, k*137)
		for i := range bf {
			bf[i] = uint16(math.Float32bits(float32(rng.NormFloat64()*0.05)) >> 16)
		}
		w, _ := NewWeightsF16(k, 137)
		if _, err := w.PackBF16(bf); err != nil {
			t.Fatal(err)
		}
		want, got := make([]float32, 137), make([]float32, 137)
		if err := MulInto(want, x, 1, w, normal); err != nil {
			t.Fatal(err)
		}
		run := func() {
			if err := compact.PrepareRowF16(k); err != nil {
				panic(err)
			}
			compact.SetRowScale(0, MaxAbs(x))
			if err := compact.PackRange(x, k, 0, k); err != nil {
				panic(err)
			}
			if err := MulPanels(got, 137, compact, w, 0, w.Panels(), nil); err != nil {
				panic(err)
			}
		}
		run()
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("K=%d column=%d changed output", k, i)
			}
		}
		if n := testing.AllocsPerRun(10, run); n != 0 {
			t.Fatalf("allocations=%g", n)
		}
		// The same workspace must safely return to a multi-row tile after decode.
		two := append(slices.Clone(x), x...)
		if err := compact.Pack(two, 2, k); err != nil {
			t.Fatal(err)
		}
		if compact.compactRow {
			t.Fatal("general Prepare retained compact mode")
		}
	}
}

func TestCompactRowRejectsUnsupportedShapes(t *testing.T) {
	ws, _ := NewWorkspace(32)
	for _, k := range []int{-16, 0, 1, 15, 17, 48} {
		if err := ws.PrepareRowF16(k); err != ErrDimensions {
			t.Fatalf("accepted K=%d", k)
		}
	}
	if !Available() {
		if err := ws.PrepareRowF16(16); err != ErrDimensions {
			t.Fatal("accepted compact row without SME")
		}
		return
	}
	if err := ws.PrepareRowF16(16); err != nil {
		t.Fatal(err)
	}
	w, _ := NewWeights(16, 64)
	if err := MulPanels(make([]float32, 64), 64, ws, w, 0, 1, nil); err != ErrDimensions {
		t.Fatal("accepted Q8 weights for a compact F16 tile")
	}
}

// BenchmarkF16RowPacking isolates the representation conversion used by one
// decoder row. Both variants feed the identical SME row multiplication.
func BenchmarkF16RowPacking(b *testing.B) {
	if !Available() {
		b.Skip("compact row requires SME")
	}
	for _, k := range []int{2048, 6144} {
		for _, compact := range []bool{false, true} {
			mode := "tile"
			if compact {
				mode = "compact"
			}
			b.Run(fmt.Sprintf("%d/%s", k, mode), func(b *testing.B) {
				ws, err := NewWorkspace(k)
				if err != nil {
					b.Fatal(err)
				}
				if compact {
					err = ws.PrepareRowF16(k)
				} else {
					err = ws.Prepare(1, k)
				}
				if err != nil {
					b.Fatal(err)
				}
				x := make([]float32, k)
				for i := range x {
					x[i] = float32(i%37-18) / 32
				}
				ws.SetRowScale(0, MaxAbs(x))
				if err = ws.PackRange(x, k, 0, k); err != nil {
					b.Fatal(err)
				}
				want := ws.row[k-1]
				b.ReportAllocs()
				for b.Loop() {
					if err = ws.PackRange(x, k, 0, k); err != nil {
						b.Fatal(err)
					}
				}
				if ws.row[k-1] != want {
					b.Fatal("packed row changed")
				}
			})
		}
	}
}
