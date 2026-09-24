// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import (
	"math"
	"math/rand"
	"testing"
)

func int8Fixture(t testing.TB, k, n int, seed int64) (*WeightsI8, []int8, []float32) {
	rng := rand.New(rand.NewSource(seed))
	q := make([]int8, k*n)
	for i := range q {
		q[i] = int8(rng.Intn(255) - 127)
	}
	scales := make([]float32, n)
	for i := range scales {
		scales[i] = float32(rng.Intn(100)+1) / 997
	}
	w, err := NewWeightsI8(k, n)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Pack(q, scales); err != nil {
		t.Fatal(err)
	}
	return w, q, scales
}

func packI8(t testing.TB, ws *WorkspaceI8, x []float32, rows, k int) {
	if err := ws.Prepare(rows, k); err != nil {
		t.Fatal(err)
	}
	for r := range rows {
		ws.SetRowScale(r, MaxAbs(x[r*k:(r+1)*k]))
	}
	if err := ws.PackRange(x, k, 0, k); err != nil {
		t.Fatal(err)
	}
}

// TestInt8KernelMatchesExactOracle checks the SME kernel against the integer
// oracle bit for bit, and the NEON quantizer against scalar rounding.
func TestInt8KernelMatchesExactOracle(t *testing.T) {
	for _, shape := range [][3]int{{1, 4, 1}, {3, 7, 17}, {12, 64, 64}, {16, 257, 129}, {12, 4096, 1024}, {5, 12288, 64}} {
		rows, k, n := shape[0], shape[1], shape[2]
		t.Run(shapeName(rows, k, n), func(t *testing.T) {
			w, q, scales := int8Fixture(t, k, n, int64(k*31+n))
			rng := rand.New(rand.NewSource(int64(rows + k)))
			x := make([]float32, rows*k)
			for i := range x {
				x[i] = float32(rng.NormFloat64()) * float32(1+i%5)
			}
			x[0] = 0 // exercise an exact zero
			ws, _ := NewWorkspaceI8(k)
			packI8(t, ws, x, rows, k)
			for r := range rows {
				inv := ws.rowInverse[r]
				for kk := range k {
					got := ws.activation[(kk/4)*4*ActivationRows+r*4+kk%4]
					if want := quantize(x[r*k+kk] * inv); got != want {
						t.Fatalf("row %d col %d: packed %d, want %d", r, kk, got, want)
					}
				}
			}
			got := make([]float32, rows*n)
			if err := MulPanelsI8(got, n, ws, w, 0, w.Panels()); err != nil {
				t.Fatal(err)
			}
			want := make([]float32, rows*n)
			scalarMulPanelsI8(want, n, ws, w, 0, w.Panels())
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("output %d: kernel %g, exact oracle %g", i, got[i], want[i])
				}
			}
			// Against float64 x·w, the only error is activation rounding:
			// at most half a step per element.
			for r := range rows {
				for c := range n {
					var exact, bound float64
					for kk := range k {
						wv := float64(q[c*k+kk]) * float64(scales[c])
						exact += float64(x[r*k+kk]) * wv
						bound += 0.5 * float64(ws.rowScale[r]) * math.Abs(wv)
					}
					if d := math.Abs(float64(got[r*n+c]) - exact); d > bound*1.001+1e-4 {
						t.Fatalf("[%d,%d] = %g, exact %g, error %g > %g", r, c, got[r*n+c], exact, d, bound)
					}
				}
			}
			if allocs := testing.AllocsPerRun(5, func() {
				_ = ws.Prepare(rows, k)
				for r := range rows {
					ws.SetRowScale(r, MaxAbs(x[r*k:(r+1)*k]))
				}
				_ = ws.PackRange(x, k, 0, k)
				_ = MulPanelsI8(got, n, ws, w, 0, w.Panels())
			}); allocs != 0 {
				t.Fatalf("int8 path allocated %.1f times", allocs)
			}
		})
	}
}

func BenchmarkInt8VsF16Kernel(b *testing.B) {
	const rows, k, n = 16, 4096, 12288
	w8, _, _ := int8Fixture(b, k, n, 1)
	x := make([]float32, rows*k)
	for i := range x {
		x[i] = float32(i%13) - 6
	}
	ws8, _ := NewWorkspaceI8(k)
	packI8(b, ws8, x, rows, k)
	dst := make([]float32, rows*n)
	b.Run("int8", func(b *testing.B) {
		for b.Loop() {
			_ = MulPanelsI8(dst, n, ws8, w8, 0, w8.Panels())
		}
		b.ReportMetric(float64(rows*k*n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
	bf := make([]uint16, k*n)
	for i := range bf {
		bf[i] = uint16(math.Float32bits(float32(i%255-127)/100) >> 16)
	}
	w16, _ := NewWeightsF16(k, n)
	if _, err := w16.PackBF16(bf); err != nil {
		b.Fatal(err)
	}
	ws16, _ := NewWorkspace(k)
	if err := ws16.Pack(x, rows, k); err != nil {
		b.Fatal(err)
	}
	b.Run("f16", func(b *testing.B) {
		for b.Loop() {
			_ = MulPanels(dst, n, ws16, w16, 0, w16.Panels(), nil)
		}
		b.ReportMetric(float64(rows*k*n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}
