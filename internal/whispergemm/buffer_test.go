// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"math"
	"slices"
	"testing"
)

func TestPackedBufferRowUpdates(t *testing.T) {
	for _, n := range []int{0, 1, 15, 16, 17, 33} {
		const capacity = 520
		size, _ := PackedLen(capacity, n)
		storage := make([]float32, size+2)
		storage[0], storage[size+1] = 123, -321
		b, err := NewPackedBBuffer(capacity, n, storage[1:size+1:size+1])
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Reshape(0, n); err != nil {
			t.Fatal(err)
		}
		raw := make([]float32, capacity*n)
		for step, span := range [][2]int{{0, 1}, {1, 16}, {17, 238}, {255, 1}, {256, 1}, {7, 9}, {16, 333}, {0, 520}, {0, 0}, {0, 33}} {
			first, rows := span[0], span[1]
			input := make([]float32, rows*(n+3))
			for row := range rows {
				for col := range n {
					v := float32(step*10000 + row*100 + col)
					if (row+col)%19 == 0 {
						v = math.Float32frombits(0x80000000)
					}
					if (row+col)%41 == 0 {
						v = math.Float32frombits(0x7fc00001)
					}
					input[row*(n+3)+col], raw[(first+row)*n+col] = v, v
				}
			}
			if err := b.PackRows(input, n+3, first, rows); err != nil {
				t.Fatal(err)
			}
			want, _ := NewPackedB(first+rows, n)
			if err := want.Pack(raw[:(first+rows)*n], n, false); err != nil {
				t.Fatal(err)
			}
			assertPackedBits(t, b, want)
			if storage[0] != 123 || storage[size+1] != -321 {
				t.Fatal("wrote beyond bounded storage")
			}
		}
		for _, rows := range []int{32, 17, 1, 0} {
			if err := b.CopyRowsFrom(b, rows); err != nil {
				t.Fatal(err)
			}
			want, _ := NewPackedB(rows, n)
			if err := want.Pack(raw[:rows*n], n, false); err != nil {
				t.Fatal(err)
			}
			assertPackedBits(t, b, want)
		}
	}
}

func TestPackedBufferCopiesAndBounds(t *testing.T) {
	for _, shape := range [][2]int{{3, 17}, {17, 65}, {0, 0}} {
		k, n := shape[0], shape[1]
		raw := make([]float32, k*n)
		for i := range raw {
			raw[i] = float32(i%29 - 14)
		}
		src, _ := NewPackedB(k, n)
		if err := src.Pack(raw, n, false); err != nil {
			t.Fatal(err)
		}
		size, _ := PackedLen(k, n)
		dst, _ := NewPackedBBuffer(k, n, make([]float32, size))
		for _, cols := range []int{n, n / 2, 0} {
			if err := dst.CopyColumnsFrom(src, cols); err != nil {
				t.Fatal(err)
			}
			want, _ := NewPackedB(k, cols)
			if err := want.Pack(raw, n, false); err != nil {
				t.Fatal(err)
			}
			assertPackedBits(t, dst, want)
		}
		if err := dst.Reshape(k, n); err != nil {
			t.Fatal(err)
		}
		for _, rows := range []int{k, k / 2, 0} {
			if err := dst.CopyRowsFrom(src, rows); err != nil {
				t.Fatal(err)
			}
			want, _ := NewPackedB(rows, n)
			if err := want.Pack(raw[:rows*n], n, false); err != nil {
				t.Fatal(err)
			}
			assertPackedBits(t, dst, want)
		}
		before := slices.Clone(dst.data)
		if err := dst.Reshape(k+100, n+100); err != ErrShape {
			t.Fatal("borrowed storage grew")
		}
		if !slices.Equal(before, dst.data) {
			t.Fatal("failed growth changed data")
		}
	}
	for _, shape := range [][2]int{{-1, 1}, {1, -1}, {math.MaxInt, 2}, {2, math.MaxInt}} {
		if _, err := PackedLen(shape[0], shape[1]); err != ErrShape {
			t.Fatal("accepted overflowing shape")
		}
	}
	if _, err := NewPackedBBuffer(2, 17, make([]float32, 63)); err != ErrShape {
		t.Fatal("accepted short storage")
	}
}

func TestPackedBufferWarmUpdatesAllocateNothing(t *testing.T) {
	b, _ := NewPackedB(256, 33)
	src := make([]float32, 256*33)
	if err := b.Pack(src, 33, false); err != nil {
		t.Fatal(err)
	}
	if got := testing.AllocsPerRun(10, func() {
		if err := b.CopyRowsFrom(b, 127); err != nil {
			panic(err)
		}
		if err := b.PackRows(src, 33, 127, 129); err != nil {
			panic(err)
		}
	}); got != 0 {
		t.Fatalf("allocations=%g", got)
	}
}

func assertPackedBits(t *testing.T, got, want *PackedB) {
	t.Helper()
	if got.k != want.k || got.n != want.n || len(got.data) != len(want.data) {
		t.Fatal("packed shape differs")
	}
	for i, v := range want.data {
		if math.Float32bits(got.data[i]) != math.Float32bits(v) {
			t.Fatalf("packed value %d differs", i)
		}
	}
}

func TestFixedPitchPagesIgnoreUnusedRows(t *testing.T) {
	testFixedPitchPages(t)
}

func testFixedPitchPages(t *testing.T) {
	accelerated := smeEnabled
	for _, k := range []int{1, 3, 16, 17, 33, 129, 255} {
		for _, n := range []int{1, 15, 17, 31, 33, 65} {
			const capacity = 256
			size, _ := PackedLen(capacity, n)
			storage := make([]float32, size)
			for i := range storage {
				storage[i] = float32(math.NaN())
			}
			page, _ := NewPackedBRows(capacity, n, storage)
			raw := make([]float32, k*n)
			for i := range raw {
				raw[i] = float32(i%23-11) / 8
			}
			if err := page.PackRows(raw, n, 0, k); err != nil {
				t.Fatal(err)
			}
			// Appending must never relocate or overwrite a preceding value.
			before := slices.Clone(storage)
			if k < capacity {
				if err := page.PackRows(make([]float32, n), n, k, 1); err != nil {
					t.Fatal(err)
				}
				for c := 0; c < n; c += panelColumns {
					for r := 0; r < k; r++ {
						for j := 0; j < panelColumns; j++ {
							idx := c*capacity + r*panelColumns + j
							if math.Float32bits(before[idx]) != math.Float32bits(storage[idx]) {
								t.Fatal("append moved old values")
							}
						}
					}
				}
				if err := page.CopyRowsFrom(page, k); err != nil {
					t.Fatal(err)
				}
			}
			for _, rows := range []int{1, 3, 17, 32, 33} {
				a := make([]float32, rows*k)
				for i := range a {
					a[i] = float32(i%13-6) / 8
				}
				stride := n + 3
				dst := make([]float32, rows*stride)
				for i := range dst {
					dst[i] = 12345
				}
				if err := page.MulScratch(dst, stride, a, k, rows, make([]float32, ScratchLen(k))); err != nil {
					t.Fatal(err)
				}
				for r := range rows {
					for c := range n {
						var want float64
						for j := range k {
							want += float64(a[r*k+j]) * float64(raw[j*n+c])
						}
						if got := dst[r*stride+c]; math.IsNaN(float64(got)) || float64(got) != want {
							t.Fatalf("SME=%v shape=%dx%dx%d at %d,%d: %g != %g", accelerated, rows, k, n, r, c, got, want)
						}
					}
					for c := n; c < stride; c++ {
						if dst[r*stride+c] != 12345 {
							t.Fatal("output padding changed")
						}
					}
				}
			}
			copyPage, _ := NewPackedBRows(capacity+16, n, make([]float32, (capacity+16)*((n+15)/16)*16))
			if err := copyPage.CopyRowsFrom(page, k); err != nil {
				t.Fatal(err)
			}
			for c := 0; c < n; c += panelColumns {
				for r := range k {
					for j := 0; j < panelColumns; j++ {
						if storage[c*capacity+r*panelColumns+j] != copyPage.data[c*(capacity+16)+r*panelColumns+j] {
							t.Fatal("pitched copy changed values")
						}
					}
				}
			}
		}
	}
}

func TestAppendColumnsPreservesPartialPanels(t *testing.T) {
	for _, k := range []int{1, 7, 16, 17, 128} {
		for _, n := range []int{1, 15, 16, 17, 63, 129} {
			raw := make([]float32, n*k)
			for i := range raw {
				raw[i] = float32(i%31 - 15)
			}
			want, _ := NewPackedB(k, n)
			if err := want.Pack(raw, k, true); err != nil {
				t.Fatal(err)
			}
			for first := 0; first <= min(n, 33); first++ {
				got, _ := NewPackedB(k, n)
				if err := got.Reshape(k, first); err != nil {
					t.Fatal(err)
				}
				if err := got.PackColumns(raw[:first*k], k, 0); err != nil {
					t.Fatal(err)
				}
				if err := got.Reshape(k, n); err != nil {
					t.Fatal(err)
				}
				if err := got.PackColumns(raw[first*k:], k, first); err != nil {
					t.Fatal(err)
				}
				assertPackedBits(t, got, want)
			}
		}
	}
}
