// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"math"
	"sync"
	"testing"
)

func nextValue(state *uint32) float32 {
	x := *state
	x ^= x << 13
	x ^= x >> 17
	x ^= x << 5
	*state = x
	return float32(int32(x>>8)-8388608) * (1.0 / 8388608)
}

// This oracle reads the original, unpacked N-by-K weights. It shares neither
// the packing index nor the SIMD traversal/reduction with the implementation.
func reference64(a []float32, aStride int, weight []float32, weightStride, r, c, k int) (sum, sumAbs float64) {
	for p := 0; p < k; p++ {
		product := float64(a[r*aStride+p]) * float64(weight[c*weightStride+p])
		sum += product
		sumAbs += math.Abs(product)
	}
	return sum, sumAbs
}

func TestMulAgainstIndependentOracle(t *testing.T) {
	t.Logf("public dispatch: %s", kernelName)
	shapes := [][3]int{
		{0, 0, 0}, {0, 7, 9}, {3, 0, 11}, {5, 9, 0},
		{1, 1, 1}, {1, 3, 7}, {2, 4, 8}, {3, 5, 9},
		{4, 7, 15}, {5, 8, 16}, {7, 9, 17}, {8, 63, 7},
		{9, 64, 8}, {5, 65, 9}, {7, 384, 17}, {5, 1536, 9},
		{1500, 3, 8}, {1, 64, 1500},
		{4, 0, 32}, {4, 1, 32}, {5, 3, 33}, {6, 4, 63},
		{7, 5, 64}, {8, 17, 65}, {9, 65, 96}, {5, 1500, 32},
	}
	for _, shape := range shapes {
		m, k, n := shape[0], shape[1], shape[2]
		for _, offset := range []int{0, 1, 3} {
			as, ws, cs := k+3, k+5, n+7
			aStorage := make([]float32, offset+m*as)
			a := aStorage[offset:]
			weight := make([]float32, n*ws)
			normal := make([]float32, k*(n+2))
			state := uint32(0xc001cafe)
			for i := range a {
				a[i] = nextValue(&state)
			}
			for c := 0; c < n; c++ {
				for p := 0; p < k; p++ {
					weight[c*ws+p] = nextValue(&state)
					normal[p*(n+2)+c] = weight[c*ws+p]
				}
			}
			b, err := NewPackedB(k, n)
			if err != nil {
				t.Fatal(err)
			}
			for _, transposed := range []bool{false, true} {
				src, stride := normal, n+2
				if transposed {
					src, stride = weight, ws
				}
				if err := b.Pack(src, stride, transposed); err != nil {
					t.Fatal(err)
				}
				dstStorage := make([]float32, offset+m*cs+1)
				for i := range dstStorage {
					dstStorage[i] = -317
				}
				dst := dstStorage[offset : offset+m*cs]
				if err := b.Mul(dst, cs, a, as, m); err != nil {
					t.Fatal(err)
				}
				for r := 0; r < m; r++ {
					for c := 0; c < n; c++ {
						want, sumAbs := reference64(a, as, weight, ws, r, c, k)
						got := float64(dst[r*cs+c])
						// Conservative FP32 forward error bound (4*K unit
						// roundoffs), valid for cancellation as well as normal sums.
						limit := 4 * float64(k+1) * (1.0 / (1 << 24)) * sumAbs
						if math.IsNaN(got) || math.Abs(got-want) > limit {
							t.Fatalf("%s shape=%v transpose=%t offset=%d [%d,%d]: got %.9g want %.9g error %.3g > %.3g", kernelName, shape, transposed, offset, r, c, got, want, math.Abs(got-want), limit)
						}
					}
					for c := n; c < cs; c++ {
						if dst[r*cs+c] != -317 {
							t.Fatalf("output row padding overwritten: shape=%v row=%d column=%d", shape, r, c)
						}
					}
				}
				if dstStorage[len(dstStorage)-1] != -317 || (offset != 0 && dstStorage[0] != -317) {
					t.Fatal("output boundary sentinel overwritten")
				}
			}
		}
	}
}

func TestMulExactDyadicTails(t *testing.T) {
	for m := 1; m <= 9; m++ {
		for k := 1; k <= 17; k++ {
			for n := 1; n <= 35; n++ {
				a, w := make([]float32, m*k), make([]float32, n*k)
				for i := range a {
					a[i] = float32(i%19-9) / 16
				}
				for i := range w {
					w[i] = float32(i%17-8) / 8
				}
				b, _ := NewPackedB(k, n)
				if err := b.Pack(w, k, true); err != nil {
					t.Fatal(err)
				}
				dst := make([]float32, m*n)
				if err := b.Mul(dst, n, a, k, m); err != nil {
					t.Fatal(err)
				}
				for r := 0; r < m; r++ {
					for c := 0; c < n; c++ {
						want, _ := reference64(a, k, w, k, r, c, k)
						if float64(dst[r*n+c]) != want {
							t.Fatalf("exact shape [%d,%d,%d] [%d,%d]: %g want %g", m, k, n, r, c, dst[r*n+c], want)
						}
					}
				}
			}
		}
	}
}

func TestMulSpecialValues(t *testing.T) {
	for _, value := range []float32{0, float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		const m, k, n = 5, 7, 9
		a, weights := make([]float32, m*k), make([]float32, n*k)
		for i := range a {
			a[i] = 1
		}
		for i := range weights {
			weights[i] = value
		}
		b, _ := NewPackedB(k, n)
		if err := b.Pack(weights, k, true); err != nil {
			t.Fatal(err)
		}
		dst := make([]float32, m*n)
		if err := b.Mul(dst, n, a, k, m); err != nil {
			t.Fatal(err)
		}
		want := float32(float64(value) * k)
		for i, got := range dst {
			if math.IsNaN(float64(want)) && math.IsNaN(float64(got)) {
				continue
			}
			if got != want {
				t.Fatalf("special %g output[%d]=%g want %g", value, i, got, want)
			}
		}
	}
}

// Overlapping input rows read a sliding window without copying it.
func TestMulOverlappingRows(t *testing.T) {
	const m, k, n, hop = 37, 40, 33, 7
	signal := make([]float32, (m-1)*hop+k)
	state := uint32(5)
	for i := range signal {
		signal[i] = nextValue(&state)
	}
	weights := make([]float32, k*n)
	for i := range weights {
		weights[i] = nextValue(&state)
	}
	b, _ := NewPackedB(k, n)
	if err := b.Pack(weights, n, false); err != nil {
		t.Fatal(err)
	}
	frames := make([]float32, m*k)
	for r := 0; r < m; r++ {
		copy(frames[r*k:], signal[r*hop:r*hop+k])
	}
	want, got := make([]float32, m*n), make([]float32, m*n)
	if err := b.Mul(want, n, frames, k, m); err != nil {
		t.Fatal(err)
	}
	if err := b.Mul(got, n, signal, hop, m); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("index %d: %v != %v", i, got[i], want[i])
		}
	}
}

func TestValidationAndZeroAllocations(t *testing.T) {
	const maxInt = int(^uint(0) >> 1)
	for _, dims := range [][2]int{{-1, 4}, {4, -1}, {maxInt, 16}, {1, maxInt}} {
		if _, err := NewPackedB(dims[0], dims[1]); err != ErrShape {
			t.Fatalf("NewPackedB%v: %v", dims, err)
		}
	}
	var nilB *PackedB
	if nilB.Pack(nil, 0, false) != ErrNilMatrix || nilB.Mul(nil, 0, nil, 0, 0) != ErrNilMatrix {
		t.Fatal("nil receiver must return ErrNilMatrix")
	}
	b, _ := NewPackedB(17, 19)
	a, weights, dst := make([]float32, 7*17), make([]float32, 19*17), make([]float32, 7*19)
	if k, n := b.Dims(); k != 17 || n != 19 {
		t.Fatal("incorrect dimensions")
	}
	if b.Pack(weights[:len(weights)-1], 17, true) != ErrShape || b.Pack(weights, 16, true) != ErrShape {
		t.Fatal("invalid packing shape accepted")
	}
	if b.Mul(dst, 19, a[:6*17+16], 17, 7) != ErrShape || b.Mul(dst[:len(dst)-1], 19, a, 17, 7) != ErrShape || b.Mul(dst, maxInt, a, maxInt, 7) != ErrShape {
		t.Fatal("invalid multiplication shape accepted")
	}
	if allocations := testing.AllocsPerRun(20, func() {
		if err := b.Pack(weights, 17, true); err != nil {
			panic(err)
		}
		if err := b.Mul(dst, 19, a, 17, 7); err != nil {
			panic(err)
		}
	}); allocations != 0 {
		t.Fatalf("Pack + Mul allocated %g objects", allocations)
	}
}

func TestConcurrentMul(t *testing.T) {
	b, _ := NewPackedB(65, 17)
	weights, a := make([]float32, 65*17), make([]float32, 9*65)
	for i := range weights {
		weights[i] = float32(i%7 - 3)
	}
	for i := range a {
		a[i] = float32(i%5 - 2)
	}
	if err := b.Pack(weights, 65, true); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 9*17)
	if err := b.Mul(want, 17, a, 65, 9); err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	for range 4 {
		group.Go(func() {
			got := make([]float32, len(want))
			for range 5 {
				if err := b.Mul(got, 17, a, 65, 9); err != nil {
					t.Error(err)
					return
				}
				for i := range got {
					if got[i] != want[i] {
						t.Errorf("concurrent output[%d]=%g, want %g", i, got[i], want[i])
						return
					}
				}
			}
		})
	}
	group.Wait()
}

func TestPackedBReshapeReusesStorage(t *testing.T) {
	b, err := NewPackedB(64, 100)
	if err != nil {
		t.Fatal(err)
	}
	for _, shape := range [][2]int{{17, 33}, {64, 100}, {5, 1}, {64, 97}} {
		k, n := shape[0], shape[1]
		if allocs := testing.AllocsPerRun(1, func() {
			if err := b.Reshape(k, n); err != nil {
				panic(err)
			}
		}); allocs != 0 {
			t.Fatalf("reshape to %dx%d allocated", k, n)
		}
		src := make([]float32, n*k)
		for i := range src {
			src[i] = float32(i%13) - 6
		}
		if err := b.Pack(src, k, true); err != nil {
			t.Fatal(err)
		}
		a := make([]float32, 3*k)
		for i := range a {
			a[i] = float32(i%5) - 2
		}
		got := make([]float32, 3*n)
		if err := b.Mul(got, n, a, k, 3); err != nil {
			t.Fatal(err)
		}
		for r := range 3 {
			for c := range n {
				var want float32
				for kk := range k {
					want += a[r*k+kk] * src[c*k+kk]
				}
				if d := got[r*n+c] - want; d > 1e-3 || d < -1e-3 {
					t.Fatalf("%dx%d [%d,%d] = %g, want %g", k, n, r, c, got[r*n+c], want)
				}
			}
		}
	}
	if err := b.Reshape(1000, 1000); err != nil {
		t.Fatal(err)
	}
	if k, n := b.Dims(); k != 1000 || n != 1000 {
		t.Fatalf("dims %d %d", k, n)
	}
}
