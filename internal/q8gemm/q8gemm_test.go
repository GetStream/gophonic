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
			scalarMulPanels(want, n, ws, w, 0, w.panels)
			// Same exact products, different FP32 summation order.
			assertClose(t, got, want, 2e-5*math.Sqrt(float64(k)))
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
				v := float64(x[row*k+kk]) * float64(w.at(col, kk))
				sum += v
				sumAbs += math.Abs(v)
			}
			want := float32(sum * float64(w.scales[col]))
			delta := math.Abs(float64(got[row*n+col] - want))
			// FP16 activation rounding contributes at most 2^-11 of each
			// product's magnitude, plus FP32 accumulation error.
			bound := 1e-6 + (1.0/2048+1e-5)*sumAbs*math.Abs(float64(w.scales[col]))
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

func TestMulPanelsRangesAndStride(t *testing.T) {
	const rows, k, n, stride = 11, 257, 197, 211
	w, ws, x, full := fixture(t, rows, k, n, 7)
	scaleRows := func() {
		for row := range rows {
			ws.SetRowScale(row, MaxAbs(x[row*k:(row+1)*k]))
		}
	}
	if err := MulInto(full, x, rows, w, ws); err != nil {
		t.Fatal(err)
	}
	if err := ws.Prepare(rows, k); err != nil {
		t.Fatal(err)
	}
	scaleRows()
	for k0 := 0; k0 < k; k0 += 50 {
		if err := ws.PackRange(x, k, k0, min(k, k0+50)); err != nil {
			t.Fatal(err)
		}
	}
	got := make([]float32, (rows-1)*stride+n)
	for i := range got {
		got[i] = -7
	}
	for p := 0; p < w.Panels(); p++ {
		if err := MulPanels(got, stride, ws, w, p, p+1, NewScratch(k)); err != nil {
			t.Fatal(err)
		}
	}
	for row := range rows {
		assertClose(t, got[row*stride:row*stride+n], full[row*n:(row+1)*n], 1e-6)
		if row+1 < rows {
			for _, v := range got[row*stride+n : (row+1)*stride] {
				if v != -7 {
					t.Fatalf("row %d padding overwritten: %g", row, v)
				}
			}
		}
	}
}

func TestF16Conversion(t *testing.T) {
	// Every FP16 value round-trips, and conversion rounds to nearest even.
	for h := range 1 << 16 {
		f := f16ToF32(uint16(h))
		if f != f {
			continue
		}
		if got := f32ToF16(f); got != uint16(h) && !(f == 0 && got&0x7fff == 0) {
			t.Fatalf("f16 %#04x -> %g -> %#04x", h, f, got)
		}
	}
	rng := rand.New(rand.NewSource(3))
	for range 200000 {
		f := float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(40)-25)))
		h := f32ToF16(f)
		got := float64(f16ToF32(h))
		// No other FP16 value is closer.
		for _, other := range []uint16{h - 1, h + 1} {
			if o := float64(f16ToF32(other)); o == o && math.Abs(o-float64(f)) < math.Abs(got-float64(f)) {
				t.Fatalf("%g rounded to %g, but %g is closer", f, got, o)
			}
		}
	}
}

func TestRowScalingHandlesLargeAndTinyRows(t *testing.T) {
	const k, n = 96, 64
	w, ws, x, got := fixture(t, 3, k, n, 11)
	for i := range k {
		x[i] *= 1e6     // would overflow FP16 unscaled
		x[k+i] *= 1e-7  // would be FP16 subnormal or zero unscaled
		x[2*k+i] *= 0.5 // ordinary
	}
	if err := MulInto(got, x, 3, w, ws); err != nil {
		t.Fatal(err)
	}
	for row := range 3 {
		for col := range n {
			var sum, sumAbs float64
			for kk := range k {
				v := float64(x[row*k+kk]) * float64(w.at(col, kk))
				sum += v
				sumAbs += math.Abs(v)
			}
			want := sum * float64(w.scales[col])
			if d := math.Abs(float64(got[row*n+col]) - want); d > (1.0/2048)*sumAbs*float64(w.scales[col]) {
				t.Fatalf("row %d col %d: got %g want %g", row, col, got[row*n+col], want)
			}
		}
	}
}

func TestNEONPackMatchesScalarConversion(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, k := range []int{1, 2, 7, 8, 9, 16, 31, 33, 100, 4096} {
		for _, rows := range []int{1, 5, 16} {
			x := make([]float32, rows*k)
			for i := range x {
				x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(30)-15)))
			}
			x[rng.Intn(len(x))] = 7e5 // force a downscaled row
			ws, _ := NewWorkspace(k)
			if err := ws.Pack(x, rows, k); err != nil {
				t.Fatal(err)
			}
			for row := range rows {
				var want float32
				for _, v := range x[row*k : (row+1)*k] {
					want = max(want, float32(math.Abs(float64(v))))
				}
				if got := MaxAbs(x[row*k : (row+1)*k]); got != want {
					t.Fatalf("k=%d row %d MaxAbs=%g want %g", k, row, got, want)
				}
				for kk := range k {
					got := ws.activation[(kk/2)*2*ActivationRows+row*2+kk%2]
					if want := f32ToF16(x[row*k+kk] * ws.rowScale[row]); got != want {
						t.Fatalf("k=%d row %d col %d: packed %#04x want %#04x", k, row, kk, got, want)
					}
				}
			}
			for pair := range (k + 1) / 2 {
				for _, v := range ws.activation[pair*2*ActivationRows+rows*2 : (pair+1)*2*ActivationRows] {
					if v != 0 {
						t.Fatalf("k=%d rows=%d pair %d has nonzero padding", k, rows, pair)
					}
				}
			}
			if k%2 == 1 {
				for row := range rows {
					if v := ws.activation[(k/2)*2*ActivationRows+row*2+1]; v != 0 {
						t.Fatalf("odd K padding not zero: %#04x", v)
					}
				}
			}
		}
	}
	if !math.IsNaN(float64(MaxAbs([]float32{1, 2, float32(math.NaN()), 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}))) {
		t.Fatal("MaxAbs dropped a NaN")
	}
}

func TestF16WeightsExactBF16(t *testing.T) {
	for _, shape := range [][3]int{{1, 7, 5}, {12, 65, 129}, {16, 256, 64}, {12, 4096, 1024}} {
		rows, k, n := shape[0], shape[1], shape[2]
		t.Run(shapeName(rows, k, n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(k + n)))
			bf := make([]uint16, k*n)
			for i := range bf {
				// Typical checkpoint magnitudes, with one large outlier per row.
				v := float32(rng.NormFloat64() * 0.02)
				if i%k == 3%k {
					v = 1.5
				}
				bf[i] = uint16(math.Float32bits(v) >> 16)
			}
			w, err := NewWeightsF16(k, n)
			if err != nil {
				t.Fatal(err)
			}
			rounded, err := w.PackBF16(bf)
			if err != nil {
				t.Fatal(err)
			}
			if rounded != 0 {
				t.Fatalf("%d BF16 weights were rounded", rounded)
			}
			for row := range n {
				for kk := range k {
					if got, want := w.at(row, kk)*w.scales[row], BF16ToF32(bf[row*k+kk]); got != want {
						t.Fatalf("weight [%d,%d] = %g, want exact %g", row, kk, got, want)
					}
				}
			}
			ws, _ := NewWorkspace(k)
			x := make([]float32, rows*k)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}
			got := make([]float32, rows*n)
			if err := MulInto(got, x, rows, w, ws); err != nil {
				t.Fatal(err)
			}
			want := make([]float32, rows*n)
			scalarMulPanels(want, n, ws, w, 0, w.panels)
			for row := range rows {
				for col := range n {
					var exact, sumAbs float64
					for kk := range k {
						v := float64(x[row*k+kk]) * float64(BF16ToF32(bf[col*k+kk]))
						exact += v
						sumAbs += math.Abs(v)
					}
					i := row*n + col
					if d := math.Abs(float64(got[i]) - exact); d > 1e-6+(1.0/2048+1e-5)*sumAbs {
						t.Fatalf("[%d,%d] got %g, exact %g", row, col, got[i], exact)
					}
					if d := math.Abs(float64(got[i] - want[i])); d > 1e-5*math.Max(1, sumAbs) {
						t.Fatalf("[%d,%d] SME %g differs from scalar %g", row, col, got[i], want[i])
					}
				}
			}
		})
	}
}

// TestPortableKernelMatchesSME runs the no-SME path on this machine and
// checks it against the exact oracle for both weight formats.
func TestPortableKernelMatchesSME(t *testing.T) {
	forcePortable = true
	defer func() { forcePortable = false }()
	for _, f16 := range []bool{false, true} {
		for _, shape := range [][3]int{{1, 7, 5}, {12, 65, 129}, {16, 256, 200}} {
			rows, k, n := shape[0], shape[1], shape[2]
			rng := rand.New(rand.NewSource(int64(k * n)))
			var w *Weights
			if f16 {
				bf := make([]uint16, k*n)
				for i := range bf {
					bf[i] = uint16(math.Float32bits(float32(rng.NormFloat64()*0.05)) >> 16)
				}
				w, _ = NewWeightsF16(k, n)
				if _, err := w.PackBF16(bf); err != nil {
					t.Fatal(err)
				}
			} else {
				w, _, _, _ = fixture(t, rows, k, n, int64(k+n))
			}
			ws, _ := NewWorkspace(k)
			if ws.act32 == nil || NewScratch(k) == nil {
				t.Fatal("portable buffers were not allocated")
			}
			x := make([]float32, rows*k)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}
			got := make([]float32, rows*n)
			if err := MulInto(got, x, rows, w, ws); err != nil {
				t.Fatal(err)
			}
			want := make([]float32, rows*n)
			scalarMulPanels(want, n, ws, w, 0, w.panels)
			for i := range got {
				if d := math.Abs(float64(got[i] - want[i])); d > 1e-5*math.Max(1, math.Abs(float64(want[i]))*float64(k)) {
					t.Fatalf("f16=%v %v: [%d] portable %g, oracle %g", f16, shape, i, got[i], want[i])
				}
			}
			if allocs := testing.AllocsPerRun(5, func() {
				if err := MulInto(got, x, rows, w, ws); err != nil {
					panic(err)
				}
			}); allocs != 0 {
				t.Fatalf("portable MulInto allocated %.1f times", allocs)
			}
		}
	}
}
