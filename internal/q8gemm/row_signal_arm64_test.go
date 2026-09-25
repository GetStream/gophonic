// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package q8gemm

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"runtime"
	"runtime/pprof"
	"sync"
	"testing"
)

type rowSignalLane struct {
	mul       func(first, last int) error
	panels    int
	got, want []float32
}

// Signals on Darwin can clear the upper streaming-vector lanes. Exercise
// both one-panel decoder calls and multi-panel calls with private activation
// workspaces while profiling, GC, and oversubscription interrupt the kernels.
func TestF16RowSignalStorm(t *testing.T) {
	if !Available() || testing.Short() {
		t.Skip("SME signal stress")
	}
	const k, n = 2048, 509
	rng := rand.New(rand.NewSource(19))
	bf := make([]uint16, k*n)
	for i := range bf {
		bf[i] = uint16(math.Float32bits(float32(rng.NormFloat64()*0.05)) >> 16)
	}
	w, err := NewWeightsF16(k, n)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.PackBF16(bf); err != nil {
		t.Fatal(err)
	}
	lanes := make([]rowSignalLane, 2*runtime.GOMAXPROCS(0)+3)
	for g := range lanes {
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64()) * float32(math.Ldexp(1, (g%3-1)*20))
		}
		ws, err := NewWorkspace(k)
		if err != nil {
			t.Fatal(err)
		}
		if err := ws.PrepareRowF16(k); err != nil {
			t.Fatal(err)
		}
		ws.SetRowScale(0, MaxAbs(x))
		if err := ws.PackRange(x, k, 0, k); err != nil {
			t.Fatal(err)
		}
		want := make([]float32, n)
		if err := MulPanels(want, n, ws, w, 0, w.Panels(), nil); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, n)
		lanes[g] = rowSignalLane{
			mul:    func(first, last int) error { return MulPanels(got, n, ws, w, first, last, nil) },
			panels: w.Panels(), got: got, want: want,
		}
	}
	runRowSignalStorm(t, lanes)
}

func TestI8RowSignalStorm(t *testing.T) {
	if !Available() || testing.Short() {
		t.Skip("SME signal stress")
	}
	const k, n = 2048, 509
	w, _, _ := int8Fixture(t, k, n, 19)
	rng := rand.New(rand.NewSource(23))
	lanes := make([]rowSignalLane, 2*runtime.GOMAXPROCS(0)+3)
	for g := range lanes {
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64()) * float32(math.Ldexp(1, (g%3-1)*20))
		}
		ws, err := NewWorkspaceI8(k)
		if err != nil {
			t.Fatal(err)
		}
		packI8(t, ws, x, 1, k)
		want, got := make([]float32, n), make([]float32, n)
		scalarMulPanelsI8(want, n, ws, w, 0, w.Panels())
		lanes[g] = rowSignalLane{
			mul:    func(first, last int) error { return MulPanelsI8(got, n, ws, w, first, last) },
			panels: w.Panels(), got: got, want: want,
		}
	}
	runRowSignalStorm(t, lanes)
}

func runRowSignalStorm(t *testing.T, lanes []rowSignalLane) {
	t.Helper()
	// A caller running go test -cpuprofile already supplies profiling signals.
	if err := pprof.StartCPUProfile(io.Discard); err == nil {
		defer pprof.StopCPUProfile()
	}
	before := Retries()
	stop, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
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
	errs := make(chan string, len(lanes))
	for g := range lanes {
		wg.Go(func() {
			l := &lanes[g]
			for it := range 100 {
				step := l.panels
				if (it+g)%2 == 0 {
					step = 1
				}
				for first := 0; first < l.panels; first += step {
					if err := l.mul(first, min(first+step, l.panels)); err != nil {
						errs <- err.Error()
						return
					}
				}
				for i, v := range l.got {
					if math.Float32bits(v) != math.Float32bits(l.want[i]) {
						errs <- fmt.Sprintf("lane=%d iteration=%d panels/call=%d column=%d: %08x != %08x", g, it, step, i, math.Float32bits(v), math.Float32bits(l.want[i]))
						return
					}
				}
			}
		})
	}
	wg.Wait()
	close(stop)
	<-stopped
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	t.Logf("row panels recomputed after signals: %d", Retries()-before)
}
