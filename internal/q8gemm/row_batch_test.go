// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestExactF16RowBatch(t *testing.T) {
	if !Available() {
		t.Skip("exact row batch requires SME")
	}
	for _, shape := range [][3]int{{2, 16, 1}, {3, 96, 65}, {8, 256, 137}, {16, 2048, 509}, {3, 6144, 129}} {
		rows, k, n := shape[0], shape[1], shape[2]
		t.Run(fmt.Sprintf("%dx%dx%d", rows, k, n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(rows * k * n)))
			bf := make([]uint16, k*n)
			for i := range bf {
				bf[i] = uint16(math.Float32bits(float32(rng.NormFloat64()*.05)) >> 16)
			}
			w, err := NewWeightsF16(k, n)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = w.PackBF16(bf); err != nil {
				t.Fatal(err)
			}
			inputStride := k + 5
			x := make([]float32, rows*inputStride+1)[1:]
			for row := range rows {
				for col := range k {
					x[row*inputStride+col] = float32(rng.NormFloat64()) * float32(math.Ldexp(1, (row%3-1)*40))
				}
				x[row*inputStride] = math.Float32frombits(0x80000000)
			}
			if rows > 2 {
				clear(x[inputStride : inputStride+k])
			}
			stride := n + 7
			want, got := make([]float32, rows*stride), make([]float32, rows*stride)
			one, _ := NewWorkspace(k)
			batch, _ := NewWorkspace(k)
			for row := range rows {
				if err := one.PrepareRowF16(k); err != nil {
					t.Fatal(err)
				}
				one.SetRowScale(0, MaxAbs(x[row*inputStride:row*inputStride+k]))
				if err := one.PackRange(x[row*inputStride:], inputStride, 0, k); err != nil {
					t.Fatal(err)
				}
				if err := MulPanels(want[row*stride:], stride, one, w, 0, w.Panels(), nil); err != nil {
					t.Fatal(err)
				}
			}
			if err := batch.PrepareRowsF16(rows, k); err != nil {
				t.Fatal(err)
			}
			for row := range rows {
				batch.SetRowScale(row, MaxAbs(x[row*inputStride:row*inputStride+k]))
			}
			for first := 0; first < k; first += 18 {
				if err := batch.PackRange(x, inputStride, first, min(first+18, k)); err != nil {
					t.Fatal(err)
				}
			}
			for _, step := range []int{1, w.Panels()} {
				clear(got)
				for p := 0; p < w.Panels(); p += step {
					if err := MulPanels(got, stride, batch, w, p, min(p+step, w.Panels()), nil); err != nil {
						t.Fatal(err)
					}
				}
				for i, v := range got {
					if math.Float32bits(v) != math.Float32bits(want[i]) {
						t.Fatalf("step%d index%d: %08x != %08x", step, i, math.Float32bits(v), math.Float32bits(want[i]))
					}
				}
			}
			if n := testing.AllocsPerRun(1, func() {
				if err := MulPanels(got, stride, batch, w, 0, w.Panels(), nil); err != nil {
					panic(err)
				}
			}); n != 0 {
				t.Fatalf("allocs=%g", n)
			}
			if err := batch.Prepare(2, k); err != nil {
				t.Fatal(err)
			}
			if batch.compactRow || batch.compactRows {
				t.Fatal("general preparation retained compact layout")
			}
		})
	}
}

func TestExactF16RowBatchRejectsUnsupportedShapes(t *testing.T) {
	ws, _ := NewWorkspace(32)
	for _, shape := range [][2]int{{0, 16}, {17, 16}, {2, 0}, {2, -16}, {2, 15}, {2, 48}} {
		if err := ws.PrepareRowsF16(shape[0], shape[1]); err != ErrDimensions {
			t.Fatalf("shape%v error%v", shape, err)
		}
	}
	if !Available() {
		if err := ws.PrepareRowsF16(2, 16); err != ErrDimensions {
			t.Fatal("accepted batch without SME")
		}
		return
	}
	if err := ws.PrepareRowsF16(2, 16); err != nil {
		t.Fatal(err)
	}
	w, _ := NewWeights(16, 64)
	if err := MulPanels(make([]float32, 128), 64, ws, w, 0, 1, nil); err != ErrDimensions {
		t.Fatal("accepted Q8 weights for contiguous F16 rows")
	}
}

func TestExactF16RowBatchExceptionalValues(t *testing.T) {
	if !Available() {
		t.Skip("exact row batch requires SME")
	}
	const k, n = 32, 67
	cases := [][2]float32{
		{0, math.Float32frombits(1 << 31)},
		{float32(math.Inf(1)), 1},
		{float32(math.Inf(-1)), 1},
		{float32(math.Inf(1)), float32(math.Inf(-1))},
		{math.Float32frombits(0x7fc12345), -1},
		{math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32},
		{math.MaxFloat32, -math.MaxFloat32},
	}
	bf := make([]uint16, k*n)
	for i := range bf {
		bf[i] = 0x3f80
	}
	w, err := NewWeightsF16(k, n)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = w.PackBF16(bf); err != nil {
		t.Fatal(err)
	}
	x := make([]float32, len(cases)*k)
	for row, pair := range cases {
		for col := range k {
			x[row*k+col] = pair[col%2]
		}
	}
	one, _ := NewWorkspace(k)
	batch, _ := NewWorkspace(k)
	want, got := make([]float32, len(cases)*n), make([]float32, len(cases)*n)
	for row := range cases {
		if err := one.PrepareRowF16(k); err != nil {
			t.Fatal(err)
		}
		one.SetRowScale(0, MaxAbs(x[row*k:(row+1)*k]))
		if err := one.PackRange(x[row*k:], k, 0, k); err != nil {
			t.Fatal(err)
		}
		if err := MulPanels(want[row*n:], n, one, w, 0, w.Panels(), nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.PrepareRowsF16(len(cases), k); err != nil {
		t.Fatal(err)
	}
	for row := range cases {
		batch.SetRowScale(row, MaxAbs(x[row*k:(row+1)*k]))
	}
	if err := batch.PackRange(x, k, 0, k); err != nil {
		t.Fatal(err)
	}
	if err := MulPanels(got, n, batch, w, 0, w.Panels(), nil); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			t.Fatalf("row%d col%d: %08x != %08x", i/n, i%n, math.Float32bits(v), math.Float32bits(want[i]))
		}
	}
}
