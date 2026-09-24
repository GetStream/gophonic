// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64

package whispergemm

import (
	"io"
	"math"
	"runtime"
	"runtime/pprof"
	"sync"
	"testing"
)

// disableSME routes a test through the NEON or scalar kernels.
func disableSME(t *testing.T) {
	t.Helper()
	saved := smeEnabled
	smeEnabled = false
	t.Cleanup(func() { smeEnabled = saved })
}

func requireSME(t *testing.T) {
	t.Helper()
	if !smeEnabled {
		t.Skip("SME with 512-bit streaming vectors is unavailable")
	}
}

// sequentialFMA is the SME contract: every output starts at +0 and adds each
// product in increasing K order with one fused FP32 rounding. The Go arm64
// compiler fuses float32 x*y+z into FMADDS.
func sequentialFMA(dst []float32, dstStride int, a []float32, aStride int, weights []float32, k, m, n int) {
	for r := 0; r < m; r++ {
		for c := 0; c < n; c++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum = a[r*aStride+p]*weights[c*k+p] + sum
			}
			dst[r*dstStride+c] = sum
		}
	}
}

type smeCase struct {
	b       *PackedB
	a, w    []float32
	m, k, n int
	as, ds  int
}

func newSMECase(t testing.TB, m, k, n int, seed uint32) smeCase {
	as, ds := k+3, n+5
	a := make([]float32, m*as+1)[1:]
	w := make([]float32, n*k)
	for i := range a {
		a[i] = nextValue(&seed)
	}
	for i := range w {
		w[i] = nextValue(&seed)
	}
	b, err := NewPackedB(k, n)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Pack(w, k, true); err != nil {
		t.Fatal(err)
	}
	return smeCase{b: b, a: a, w: w, m: m, k: k, n: n, as: as, ds: ds}
}

func (c smeCase) output() []float32 {
	out := make([]float32, c.m*c.ds)
	for i := range out {
		out[i] = -317
	}
	return out
}

func (c smeCase) check(t testing.TB, got []float32) {
	want := make([]float32, len(got))
	for i := range want {
		want[i] = -317
	}
	sequentialFMA(want, c.ds, c.a, c.as, c.w, c.k, c.m, c.n)
	for i := range got {
		if g, w := got[i], want[i]; g != g && w != w {
			continue // FMOPA returns the default NaN rather than propagating payloads.
		}
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("m=%d k=%d n=%d index %d: %x != %x", c.m, c.k, c.n, i,
				math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func TestSMESequentialFMA(t *testing.T) {
	requireSME(t)
	for _, shape := range [][3]int{
		{1, 1, 1}, {1, 3, 7}, {2, 4, 8}, {3, 5, 9}, {4, 7, 15}, {5, 8, 16},
		{7, 9, 17}, {8, 63, 7}, {15, 16, 31}, {16, 17, 32}, {17, 15, 33},
		{31, 64, 48}, {32, 65, 64}, {33, 1, 96}, {63, 240, 384}, {64, 384, 65},
		{65, 1499, 32}, {32, 1500, 64}, {20, 1536, 384}, {100, 384, 1500},
	} {
		c := newSMECase(t, shape[0], shape[1], shape[2], 0x5eed1234)
		got := c.output()
		if err := c.b.Mul(got, c.ds, c.a, c.as, c.m); err != nil {
			t.Fatal(err)
		}
		c.check(t, got)
		if allocs := testing.AllocsPerRun(3, func() {
			if err := c.b.Mul(got, c.ds, c.a, c.as, c.m); err != nil {
				panic(err)
			}
		}); allocs != 0 {
			t.Fatalf("shape %v allocations: %g", shape, allocs)
		}
	}
}

func TestSMESpecialValues(t *testing.T) {
	requireSME(t)
	const m, k, n = 37, 65, 50
	values := []float32{0, math.Float32frombits(0x80000000), 1, -1, 1e20, -1e20, 1e-20, -1e-20, math.SmallestNonzeroFloat32}
	c := newSMECase(t, m, k, n, 1)
	for i := range c.a {
		c.a[i] = values[i%len(values)]
	}
	for i := range c.w {
		c.w[i] = values[(i/7)%len(values)]
	}
	if err := c.b.Pack(c.w, k, true); err != nil {
		t.Fatal(err)
	}
	got := c.output()
	for _, special := range []float32{0, float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc12345)} {
		for _, row := range []int{0, 5, 17, 36} {
			c.a[row*c.as+17] = special
		}
		if err := c.b.Mul(got, c.ds, c.a, c.as, m); err != nil {
			t.Fatal(err)
		}
		c.check(t, got)
	}
}

// Darwin clears the upper Z-register lanes when a signal handler returns to a
// thread in streaming mode. Profiling signals, GC preemption, and
// oversubscribed goroutines must still produce exact results.
func TestSMESignalStorm(t *testing.T) {
	requireSME(t)
	if testing.Short() {
		t.Skip("signal stress")
	}
	if err := pprof.StartCPUProfile(io.Discard); err == nil {
		defer pprof.StopCPUProfile()
	}
	c := newSMECase(t, 96, 1536, 384, 7)
	want := c.output()
	if err := c.b.Mul(want, c.ds, c.a, c.as, c.m); err != nil {
		t.Fatal(err)
	}
	c.check(t, want)
	before := smeRetries.Load()
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()
	var wg sync.WaitGroup
	errs := make(chan string, 64)
	for g := 0; g < 2*runtime.GOMAXPROCS(0)+3; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := c.output()
			for it := 0; it < 60; it++ {
				if err := c.b.Mul(got, c.ds, c.a, c.as, c.m); err != nil {
					errs <- err.Error()
					return
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						errs <- "mismatch under signals"
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	t.Logf("tiles recomputed after signals: %d", smeRetries.Load()-before)
}

func TestSMETransposePackMatchesScalar(t *testing.T) {
	requireSME(t)
	for _, shape := range [][2]int{{1, 1}, {15, 17}, {16, 16}, {17, 33}, {1500, 64}, {384, 384}, {33, 1500}, {7, 3}} {
		n, k := shape[0], shape[1]
		stride := k + 3
		src := make([]float32, n*stride)
		state := uint32(n*131 + k)
		for i := range src {
			src[i] = nextValue(&state)
		}
		got, _ := NewPackedB(k, n)
		for i := range got.data {
			got.data[i] = -317 // padding must be overwritten with zeros
		}
		if err := got.Pack(src, stride, true); err != nil {
			t.Fatal(err)
		}
		saved := smeEnabled
		smeEnabled = false
		want, _ := NewPackedB(k, n)
		err := want.Pack(src, stride, true)
		smeEnabled = saved
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.data {
			if math.Float32bits(got.data[i]) != math.Float32bits(want.data[i]) {
				t.Fatalf("n=%d k=%d index %d: %v != %v", n, k, i, got.data[i], want.data[i])
			}
		}
	}
}
