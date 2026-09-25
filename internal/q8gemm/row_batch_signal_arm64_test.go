// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package q8gemm

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

func TestF16RowBatchSignalStorm(t *testing.T) {
	if !Available() || testing.Short() {
		t.Skip("SME signal stress")
	}
	const rows, k, n = 3, 2048, 509
	rng := rand.New(rand.NewSource(61))
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
	lanes := make([]rowSignalLane, 2*runtime.GOMAXPROCS(0)+3)
	for g := range lanes {
		x := make([]float32, rows*k)
		for i := range x {
			x[i] = float32(rng.NormFloat64()) * float32(math.Ldexp(1, (i/k%3-1)*20))
		}
		ws, _ := NewWorkspace(k)
		one, _ := NewWorkspace(k)
		want, got := make([]float32, rows*n), make([]float32, rows*n)
		if err := ws.PrepareRowsF16(rows, k); err != nil {
			t.Fatal(err)
		}
		for row := range rows {
			values := x[row*k : (row+1)*k]
			ws.SetRowScale(row, MaxAbs(values))
			if err := one.PrepareRowF16(k); err != nil {
				t.Fatal(err)
			}
			one.SetRowScale(0, MaxAbs(values))
			if err := one.PackRange(values, k, 0, k); err != nil {
				t.Fatal(err)
			}
			if err := MulPanels(want[row*n:], n, one, w, 0, w.Panels(), nil); err != nil {
				t.Fatal(err)
			}
		}
		if err := ws.PackRange(x, k, 0, k); err != nil {
			t.Fatal(err)
		}
		lanes[g] = rowSignalLane{mul: func(first, last int) error { return MulPanels(got, n, ws, w, first, last, nil) }, panels: w.Panels(), got: got, want: want}
	}
	runRowSignalStorm(t, lanes)
}
