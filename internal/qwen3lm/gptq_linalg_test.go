// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/GetStream/gophonic/internal/q8gemm"
)

func TestGPTQInverseFactor(t *testing.T) {
	for _, n := range []int{1, 7, 128, 300} {
		rng := rand.New(rand.NewPCG(uint64(n), 3))
		// H = XᵀX/rows + I for random X: symmetric positive definite.
		rows := n + 17
		x := make([]float64, rows*n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		h := make([]float64, n*n)
		for i := range n {
			for j := range n {
				var s float64
				for r := range rows {
					s += x[r*n+i] * x[r*n+j]
				}
				h[i*n+j] = s / float64(rows)
			}
			h[i*n+i]++
		}
		h32 := make([]float32, n*n)
		for i, v := range h {
			h32[i] = float32(v)
		}
		u, err := gptqInverseFactor(h32, n)
		if err != nil {
			t.Fatal(err)
		}
		// H·(UᵀU) must be the identity, and U upper triangular.
		worst := 0.0
		for i := range n {
			for j := range n {
				if j < i && u[i*n+j] != 0 {
					t.Fatalf("n=%d: U[%d][%d] = %v below the diagonal", n, i, j, u[i*n+j])
				}
				var s float64
				for p := range n {
					var hinv float64 // (UᵀU)[p][j]
					for q := 0; q <= min(p, j); q++ {
						hinv += float64(u[q*n+p]) * float64(u[q*n+j])
					}
					s += h[i*n+p] * hinv
				}
				if i == j {
					s--
				}
				worst = max(worst, math.Abs(s))
			}
		}
		if worst > 1e-4 {
			t.Errorf("n=%d: max |H·H⁻¹ - I| = %.3g", n, worst)
		}
	}
}

// TestGPTQRoundBeatsNearest checks that GPTQ rounding gives a smaller output
// error ‖X·(W−Q)ᵀ‖ than round-to-nearest with the same scales, on inputs with
// correlated channels.
func TestGPTQRoundBeatsNearest(t *testing.T) {
	const rows, k, n = 512, 256, 16
	rng := rand.New(rand.NewPCG(5, 6))
	x := make([]float32, rows*k)
	for r := range rows {
		base := rng.NormFloat64()
		for i := range k {
			x[r*k+i] = float32(base + 0.5*rng.NormFloat64())
		}
	}
	w := make([]float32, n*k)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	outErr := func(q []float32) float64 {
		var e float64
		for r := range rows {
			for o := range n {
				var s float64
				for i := range k {
					s += float64(x[r*k+i]) * float64(w[o*k+i]-q[o*k+i])
				}
				e += s * s
			}
		}
		return e
	}
	for _, bits := range []int{8, 4} {
		// Hessian of x, damped as quantizeGroup does.
		h := make([]float32, k*k)
		var mean float64
		for i := range k {
			for j := range k {
				var s float64
				for r := range rows {
					s += float64(x[r*k+i]) * float64(x[r*k+j])
				}
				h[i*k+j] = float32(2 * s / rows)
			}
			mean += float64(h[i*k+i]) / k
		}
		for i := range k {
			h[i*k+i] += float32(gptqDamp * mean)
		}
		u, err := gptqInverseFactor(h, k)
		if err != nil {
			t.Fatal(err)
		}
		g := append([]float32(nil), w...)
		cb, sb := gptqRowBytes(bits, k)
		if err := gptqRound(g, n, k, u, bits, make([]byte, n*cb), make([]byte, n*sb)); err != nil {
			t.Fatal(err)
		}
		nearest := append([]float32(nil), w...)
		for o := range n {
			row := nearest[o*k : (o+1)*k]
			if bits == 8 {
				s := q8gemm.MaxAbs(row) / 127
				for i, v := range row {
					row[i] = float32(math.RoundToEven(float64(v/s))) * s
				}
				continue
			}
			for b := 0; b < k; b += q4GroupSize {
				d := q4GroupScale(row[b : b+q4GroupSize])
				for i := b; i < b+q4GroupSize; i++ {
					row[i] = float32(max(-8, min(7, math.RoundToEven(float64(row[i]/d))))) * d
				}
			}
		}
		ge, ne := outErr(g), outErr(nearest)
		t.Logf("bits=%d: GPTQ output error %.4g, nearest %.4g (%.2fx)", bits, ge, ne, ne/ge)
		if ge >= ne {
			t.Errorf("bits=%d: GPTQ does not beat round-to-nearest", bits)
		}
	}
}
