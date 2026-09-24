// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

func TestMulIntoRandomizedTails(t *testing.T) {
	for _, k := range []int{1, 3, 7, 16, 31, 65, 257} {
		for _, n := range []int{1, 15, 16, 17, 63, 64, 65, 129} {
			for _, rows := range []int{1, 12, 16} {
				t.Run(shapeName(rows, k, n), func(t *testing.T) {
					checkShape(t, rows, k, n)
				})
			}
		}
	}
}

func TestMulNoAllocationsAfterWarmup(t *testing.T) {
	const rows, k, n = 12, 129, 67
	w, ws, x, dst := fixture(t, rows, k, n, 21)
	if err := MulInto(dst, x, rows, w, ws); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(10, func() {
		if err := MulInto(dst, x, rows, w, ws); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warmed MulInto allocations = %g, want 0", allocs)
	}
}

func TestSMEKernelMatchesScalarOracle(t *testing.T) {
	if !usingSME() {
		if os.Getenv("Q8GEMM_REQUIRE_SME") == "1" {
			t.Fatal("Q8GEMM_REQUIRE_SME is set but SME 512-bit dispatch is inactive")
		}
		t.Skip("SME is unavailable on this target")
	}
	for _, shape := range [][3]int{{12, 4096, 4096}, {12, 4096, 1024}, {12, 4096, 12288}, {12, 12288, 4096}} {
		rows, k, n := shape[0], shape[1], shape[2]
		t.Run(shapeName(rows, k, n), func(t *testing.T) {
			w, ws, x, got := fixture(t, rows, k, n, int64(rows*100+k*10+n))
			want := make([]float32, rows*n)
			if err := MulInto(got, x, rows, w, ws); err != nil {
				t.Fatal(err)
			}
			activation := make([]float32, k*ActivationRows)
			packActivations(activation, x, rows, k)
			scalarMul(want, activation, rows, w)
			assertClose(t, got, want, 5e-5)
		})
	}
}

func checkShape(t *testing.T, rows, k, n int) {
	t.Helper()
	w, ws, x, got := fixture(t, rows, k, n, int64(rows*100+k*10+n))
	if err := MulInto(got, x, rows, w, ws); err != nil {
		t.Fatal(err)
	}
	for row := range rows {
		for col := range n {
			var sum, sumAbs float64
			for kk := range k {
				v := float64(x[row*k+kk]) * float64(w.qAt(col, kk))
				sum += v
				sumAbs += math.Abs(v)
			}
			want := float32(sum * float64(w.scales[col]))
			delta := math.Abs(float64(got[row*n+col] - want))
			bound := 5e-5 * math.Max(1, sumAbs*math.Abs(float64(w.scales[col])))
			if delta > bound {
				t.Fatalf("rows=%d K=%d N=%d at [%d,%d]: got %.9g, want %.9g, error %.3g > %.3g", rows, k, n, row, col, got[row*n+col], want, delta, bound)
			}
		}
	}
}

func fixture(t *testing.T, rows, k, n int, seed int64) (*Weights, *Workspace, []float32, []float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	q := make([]int8, k*n)
	for i := range q {
		q[i] = int8(rng.Intn(255) - 127)
	}
	scales := make([]float32, n)
	for i := range scales {
		scales[i] = float32(rng.Intn(100)+1) / 1000
	}
	w, err := NewWeights(k, n)
	if err != nil {
		t.Fatal(err)
	}
	if err = w.Pack(q, scales); err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(k)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, rows*k)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	return w, ws, x, make([]float32, rows*n)
}

func (w *Weights) qAt(row, k int) int8 {
	return w.q[(row/OutputPanel)*w.k*OutputPanel+k*OutputPanel+row%OutputPanel]
}

func assertClose(t *testing.T, got, want []float32, relativeTolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch got %d want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > relativeTolerance*math.Max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("value %d: got %.9g, want %.9g", i, got[i], want[i])
		}
	}
}

func shapeName(rows, k, n int) string {
	return "M" + itoa(rows) + "K" + itoa(k) + "N" + itoa(n)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
