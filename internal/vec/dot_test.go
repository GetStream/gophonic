// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package vec

import (
	"math"
	"testing"
)

func TestDotProduct4x4MatchesScalar(t *testing.T) {
	const width = 64
	var a [4][width]float32
	var b [4][width]float32
	for r := 0; r < 4; r++ {
		for i := 0; i < width; i++ {
			a[r][i] = float32(math.Sin(float64((r+1)*(i+3)))) * 0.2
			b[r][i] = float32(math.Cos(float64((r+2)*(i+5)))) * 0.3
		}
	}
	var got [16]float32
	Dot4x4(a[0][:], a[1][:], a[2][:], a[3][:], b[0][:], b[1][:], b[2][:], b[3][:], &got)
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			want := Dot(a[r][:], b[c][:])
			if delta := math.Abs(float64(got[r*4+c] - want)); delta > 2e-6 {
				t.Errorf("tile[%d,%d] = %.9g, want %.9g (delta %.3g)", r, c, got[r*4+c], want, delta)
			}
		}
	}
}

func TestDotKernelsMatchScalarAcrossLengths(t *testing.T) {
	// Cover zero, short, exact-vector-width, and tail lengths for both the
	// 4-wide and 8-wide vector loops. The oracle is an explicit scalar loop
	// so it stays independent of the SIMD Dot on SIMD builds.
	lengths := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 100, 128, 129, 256}
	for _, n := range lengths {
		a := make([][]float32, 4)
		b := make([][]float32, 4)
		for r := 0; r < 4; r++ {
			a[r] = make([]float32, n)
			b[r] = make([]float32, n)
			for i := 0; i < n; i++ {
				a[r][i] = float32(math.Sin(float64((r+1)*(i+3)))) * 0.2
				b[r][i] = float32(math.Cos(float64((r+2)*(i+5)))) * 0.3
			}
		}
		var got [16]float32
		Dot4x4(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], &got)
		tol := 5e-6 * (1 + float64(n)/64)
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				var want float32
				for i := 0; i < n; i++ {
					want += a[r][i] * b[c][i]
				}
				if delta := math.Abs(float64(got[r*4+c] - want)); delta > tol {
					t.Errorf("4x4 n=%d tile[%d,%d] = %.9g, want %.9g (delta %.3g > %.3g)", n, r, c, got[r*4+c], want, delta, tol)
				}
			}
		}
		s := [4]float32{}
		s[0], s[1], s[2], s[3] = Dot4(a[0], b[0], b[1], b[2], b[3])
		for c := 0; c < 4; c++ {
			var want float32
			for i := 0; i < n; i++ {
				want += a[0][i] * b[c][i]
			}
			if delta := math.Abs(float64(s[c] - want)); delta > tol {
				t.Errorf("dot4 n=%d lane %d = %.9g, want %.9g (delta %.3g > %.3g)", n, c, s[c], want, delta, tol)
			}
		}
	}
}

func TestDotKernelsExactOnSmallIntegers(t *testing.T) {
	// Small integer values keep every partial sum exactly representable, so
	// any lane mixup, dropped tail, or double count fails exactly.
	lengths := []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 100}
	for _, n := range lengths {
		a := make([][]float32, 4)
		b := make([][]float32, 4)
		for r := 0; r < 4; r++ {
			a[r] = make([]float32, n)
			b[r] = make([]float32, n)
			for i := 0; i < n; i++ {
				a[r][i] = float32((r*7+i*3)%11 - 5)
				b[r][i] = float32((r*5+i*13)%9 - 4)
			}
		}
		var got [16]float32
		Dot4x4(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], &got)
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				var want float32
				for i := 0; i < n; i++ {
					want += a[r][i] * b[c][i]
				}
				if got[r*4+c] != want {
					t.Errorf("4x4 n=%d tile[%d,%d] = %.9g, want %.9g", n, r, c, got[r*4+c], want)
				}
			}
		}
		s0, s1, s2, s3 := Dot4(a[0], b[0], b[1], b[2], b[3])
		for c, s := range [4]float32{s0, s1, s2, s3} {
			var want float32
			for i := 0; i < n; i++ {
				want += a[0][i] * b[c][i]
			}
			if s != want {
				t.Errorf("dot4 n=%d lane %d = %.9g, want %.9g", n, c, s, want)
			}
		}
	}
}

func TestDotKernelsEdgeValues(t *testing.T) {
	// NaN must poison exactly the outputs whose lanes consume it; infinities
	// and signed zeros must agree with scalar evaluation up to zero sign.
	n := 24
	mk := func(fill func(r, i int) float32) [][]float32 {
		m := make([][]float32, 4)
		for r := 0; r < 4; r++ {
			m[r] = make([]float32, n)
			for i := 0; i < n; i++ {
				m[r][i] = fill(r, i)
			}
		}
		return m
	}
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	a := mk(func(r, i int) float32 {
		if r == 1 && i == 10 {
			return nan
		}
		if r == 2 && i == 20 {
			return inf
		}
		return float32(i%5) * 0.5
	})
	b := mk(func(r, i int) float32 { return float32((i+r)%3) * 0.25 })
	var got [16]float32
	Dot4x4(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], &got)
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			var want float32
			for i := 0; i < n; i++ {
				want += a[r][i] * b[c][i]
			}
			g, w := got[r*4+c], want
			switch {
			case math.IsNaN(float64(w)):
				if !math.IsNaN(float64(g)) {
					t.Errorf("4x4 tile[%d,%d] = %.9g, want NaN", r, c, g)
				}
			case math.IsInf(float64(w), 0):
				if !math.IsInf(float64(g), 0) || math.Signbit(float64(g)) != math.Signbit(float64(w)) {
					t.Errorf("4x4 tile[%d,%d] = %.9g, want %.9g", r, c, g, w)
				}
			case g != w && !(g == 0 && w == 0):
				t.Errorf("4x4 tile[%d,%d] = %.9g, want %.9g", r, c, g, w)
			}
		}
	}
	zeros := mk(func(r, i int) float32 { return 0 })
	var gotZero [16]float32
	Dot4x4(zeros[0], zeros[1], zeros[2], zeros[3], b[0], b[1], b[2], b[3], &gotZero)
	for k, g := range gotZero {
		if g != 0 {
			t.Errorf("4x4 zero row output[%d] = %.9g, want 0", k, g)
		}
	}
}
