// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"errors"
	"math"
	"runtime"
	"sync"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

// linalgBlock is the panel width of the blocked factorizations; the
// O(n³) work runs as FP32 matrix products of this depth.
const linalgBlock = 128

// parallelRows calls f on disjoint row ranges [lo, hi) of n rows, chunk rows
// at a time, across GOMAXPROCS goroutines.
func parallelRows(n, chunk int, f func(lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), (n+chunk-1)/chunk)
	if workers <= 1 {
		for lo := 0; lo < n; lo += chunk {
			f(lo, min(lo+chunk, n))
		}
		return
	}
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		next int
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				lo := next
				next += chunk
				mu.Unlock()
				if lo >= n {
					return
				}
				f(lo, min(lo+chunk, n))
			}
		}()
	}
	wg.Wait()
}

// mulInto computes dst[m×n] (stride ds) = a[m×k] (stride as) · b, where b
// is packed K-by-N, splitting rows of a across goroutines.
func mulInto(dst []float32, ds int, a []float32, as int, m int, b *whispergemm.PackedB) error {
	var mu sync.Mutex
	var first error
	parallelRows(m, 64, func(lo, hi int) {
		if err := b.Mul(dst[lo*ds:], ds, a[lo*as:], as, hi-lo); err != nil {
			mu.Lock()
			first = errors.Join(first, err)
			mu.Unlock()
		}
	})
	return first
}

// choleskyLower replaces the lower triangle of the symmetric positive
// definite n×n matrix a (row-major) with its Cholesky factor L, a = L·Lᵀ.
// The upper triangle is left unspecified.
func choleskyLower(a []float32, n int) error {
	tmp := make([]float32, 0)
	for j0 := 0; j0 < n; j0 += linalgBlock {
		j1 := min(j0+linalgBlock, n)
		b := j1 - j0
		// Factor the diagonal block (already updated by earlier panels).
		for j := j0; j < j1; j++ {
			var d float64
			for p := j0; p < j; p++ {
				d += float64(a[j*n+p]) * float64(a[j*n+p])
			}
			d = float64(a[j*n+j]) - d
			if !(d > 0) {
				return errors.New("qwen3: GPTQ Hessian is not positive definite")
			}
			l := math.Sqrt(d)
			a[j*n+j] = float32(l)
			for i := j + 1; i < j1; i++ {
				var s float64
				for p := j0; p < j; p++ {
					s += float64(a[i*n+p]) * float64(a[j*n+p])
				}
				a[i*n+j] = float32((float64(a[i*n+j]) - s) / l)
			}
		}
		if j1 == n {
			break
		}
		// Panel below: L21 = A21·L11⁻ᵀ, row by row.
		parallelRows(n-j1, 64, func(lo, hi int) {
			for i := j1 + lo; i < j1+hi; i++ {
				row := a[i*n+j0 : i*n+j1]
				for j := range b {
					s := float64(row[j])
					for p := range j {
						s -= float64(row[p]) * float64(a[(j0+j)*n+j0+p])
					}
					row[j] = float32(s / float64(a[(j0+j)*n+j0+j]))
				}
			}
		})
		// Trailing update of the lower part: A22 -= L21·L21ᵀ.
		m := n - j1
		pb, err := whispergemm.NewPackedB(b, m)
		if err != nil {
			return err
		}
		if err := pb.Pack(a[j1*n+j0:], n, true); err != nil {
			return err
		}
		if cap(tmp) < m*m {
			tmp = make([]float32, m*m)
		}
		t := tmp[:m*m]
		if err := mulInto(t, m, a[j1*n+j0:], n, m, pb); err != nil {
			return err
		}
		parallelRows(m, 64, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				row := a[(j1+i)*n+j1 : (j1+i)*n+j1+i+1]
				for j := range row {
					row[j] -= t[i*m+j]
				}
			}
		})
	}
	return nil
}

// invertLowerTransposed returns Xᵀ for X = L⁻¹, where the lower triangle of
// l (n×n, row-major) holds L. Xᵀ is upper triangular.
func invertLowerTransposed(l []float32, n int) ([]float32, error) {
	xt := make([]float32, n*n)
	tt := make([]float32, n*linalgBlock)
	for i0 := 0; i0 < n; i0 += linalgBlock {
		i1 := min(i0+linalgBlock, n)
		b := i1 - i0
		// Tᵀ[c][r] = Σ_p X[p][c]·L[i0+r][p] over p < i0, i.e. Xᵀ[:i0,:i0]·L[i0:i1,:i0]ᵀ.
		if i0 > 0 {
			pb, err := whispergemm.NewPackedB(i0, b)
			if err != nil {
				return nil, err
			}
			if err := pb.Pack(l[i0*n:], n, true); err != nil {
				return nil, err
			}
			if err := mulInto(tt, b, xt, n, i0, pb); err != nil {
				return nil, err
			}
		}
		// Rows i0..i1 of X: X[i][c] = (δ(i,c) - Σ_{p<i} L[i][p]·X[p][c]) / L[i][i].
		parallelRows(i1, 32, func(lo, hi int) {
			for c := lo; c < hi; c++ {
				for r := max(0, c-i0); r < b; r++ {
					i := i0 + r
					var s float64
					if c < i0 {
						s = float64(tt[c*b+r])
					}
					for p := max(i0, c); p < i; p++ {
						s += float64(l[i*n+p]) * float64(xt[c*n+p])
					}
					v := -s
					if c == i {
						v = 1
					}
					xt[c*n+i] = float32(v / float64(l[i*n+i]))
				}
			}
		})
	}
	return xt, nil
}

// gptqInverseFactor returns U, upper triangular with H⁻¹ = Uᵀ·U, for
// the n×n symmetric positive definite h. h is overwritten. With J the
// reversal permutation, U = J·chol(J·H·J)⁻¹·J.
func gptqInverseFactor(h []float32, n int) ([]float32, error) {
	// A = J·H·J in place: reverse the rows and the columns.
	for i := range n / 2 {
		a, b := h[i*n:(i+1)*n], h[(n-1-i)*n:(n-i)*n]
		for j := range a {
			a[j], b[j] = b[j], a[j]
		}
	}
	for i := range n {
		row := h[i*n : (i+1)*n]
		for j := range n / 2 {
			row[j], row[n-1-j] = row[n-1-j], row[j]
		}
	}
	if err := choleskyLower(h, n); err != nil {
		return nil, err
	}
	xt, err := invertLowerTransposed(h, n)
	if err != nil {
		return nil, err
	}
	// U[r][c] = X[n-1-r][n-1-c] = Xᵀ[n-1-c][n-1-r]; write it into h.
	parallelRows(n, 64, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := h[r*n : (r+1)*n]
			for c := range n {
				if c < r {
					row[c] = 0
					continue
				}
				row[c] = xt[(n-1-c)*n+n-1-r]
			}
		}
	})
	return h, nil
}
