// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && arm64

package whispergemm

import (
	"math"
	"testing"
)

func TestWideGEMMPreservesFourRowOrder(t *testing.T) {
	for _, shape := range [][3]int{
		{4, 0, 32}, {4, 1, 32}, {5, 3, 33}, {6, 4, 63}, {7, 5, 64},
		{8, 17, 65}, {9, 65, 96}, {17, 240, 384}, {16, 384, 64},
		{16, 1024, 32}, {33, 1499, 32}, {32, 1500, 64}, {20, 1536, 384},
	} {
		m, k, n := shape[0], shape[1], shape[2]
		as, ds := k+7, n+3
		a := make([]float32, m*as+1)[1:]
		weights := make([]float32, n*k)
		state := uint32(0xbde47128)
		for i := range a {
			a[i] = nextValue(&state)
		}
		for i := range weights {
			weights[i] = nextValue(&state)
		}
		b, err := NewPackedB(k, n)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Pack(weights, k, true); err != nil {
			t.Fatal(err)
		}
		want, got := make([]float32, m*ds), make([]float32, m*ds)
		for i := range want {
			want[i], got[i] = -317, -317
		}
		mulFourRowReference(want, ds, a, as, b.data, m, k, n)
		if err := b.Mul(got, ds, a, as, m); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("shape %v index %d: %x != %x", shape, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
		if allocs := testing.AllocsPerRun(2, func() {
			if err := b.Mul(got, ds, a, as, m); err != nil {
				panic(err)
			}
		}); allocs != 0 {
			t.Fatalf("shape %v allocations: %g", shape, allocs)
		}
	}
}

func TestWideGEMMSpecialValues(t *testing.T) {
	const m, k, n = 7, 65, 64
	a, weights := make([]float32, m*k), make([]float32, n*k)
	values := []float32{0, math.Float32frombits(0x80000000), 1, -1, 1e20, -1e20, 1e-20, -1e-20, math.SmallestNonzeroFloat32}
	for i := range a {
		a[i] = values[i%len(values)]
	}
	for i := range weights {
		weights[i] = values[(i/7)%len(values)]
	}
	b, _ := NewPackedB(k, n)
	if err := b.Pack(weights, k, true); err != nil {
		t.Fatal(err)
	}
	want, got := make([]float32, m*n), make([]float32, m*n)
	for _, special := range []float32{0, float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc12345)} {
		a[17] = special
		mulFourRowReference(want, n, a, k, b.data, m, k, n)
		if err := b.Mul(got, n, a, k, m); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("special %x index %d: %x != %x", math.Float32bits(special), i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
}

// This dispatch preserves the pre-2x32 row grouping and reduction order.
// The independent FP64 oracle separately checks the numerical contract.
func mulFourRowReference(dst []float32, dstStride int, a []float32, aStride int, packed []float32, m, k, n int) {
	for c := 0; c < n; c += panelColumns {
		width := min(panelColumns, n-c)
		weights := packed[c*k : (c+panelColumns)*k]
		r := 0
		if width == panelColumns {
			for ; r+4 <= m; r += 4 {
				kernel4x16(
					a[r*aStride:r*aStride+k], a[(r+1)*aStride:(r+1)*aStride+k],
					a[(r+2)*aStride:(r+2)*aStride+k], a[(r+3)*aStride:(r+3)*aStride+k], weights,
					dst[r*dstStride+c:r*dstStride+c+width], dst[(r+1)*dstStride+c:(r+1)*dstStride+c+width],
					dst[(r+2)*dstStride+c:(r+2)*dstStride+c+width], dst[(r+3)*dstStride+c:(r+3)*dstStride+c+width],
				)
			}
		}
		for ; r < m; r++ {
			kernel1x16(a[r*aStride:r*aStride+k], weights, dst[r*dstStride+c:r*dstStride+c+width])
		}
	}
}
