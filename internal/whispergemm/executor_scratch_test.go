// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"runtime"
	"testing"
)

type scratchRowsProbe struct {
	executor *Executor
	need     int
	bad      []bool
	visits   []int
}

func (p *scratchRowsProbe) ApplyRowsScratch(scratch []float32, start, end int) {
	// These calls assign one row to each worker. Inspect ownership inside
	// the borrow rather than retaining pointers after the operation returns.
	if len(scratch) < p.need || p.need > 0 && &scratch[0] != &p.executor.scratch[start][0] {
		p.bad[start] = true
		return
	}
	for i := 0; i < p.need; i++ {
		scratch[i] = float32(start*1000 + i)
	}
	runtime.Gosched() // Give an accidental alias a chance to overwrite us.
	for i := 0; i < p.need; i++ {
		if scratch[i] != float32(start*1000+i) {
			p.bad[start] = true
		}
	}
	for i := start; i < end; i++ {
		p.visits[i]++
	}
}

func TestExecutorScratchRowsOwnershipAndReuse(t *testing.T) {
	for _, workers := range []int{1, 3, 8} {
		e, err := NewExecutor(workers)
		if err != nil {
			t.Fatal(err)
		}
		p := &scratchRowsProbe{executor: e, need: 257, bad: make([]bool, workers), visits: make([]int, workers)}
		if err := e.RowsWithScratch(p, workers, 1, p.need); err != nil {
			t.Fatal(err)
		}
		memory := e.memory
		for _, active := range []int{workers, 1, min(workers, 2), workers} {
			clear(p.visits)
			if err := e.RowsWithScratch(p, active, 1, p.need); err != nil {
				t.Fatal(err)
			}
			for i := range p.visits {
				want := 0
				if i < active {
					want = 1
				}
				if p.bad[i] || p.visits[i] != want {
					t.Fatalf("workers=%d active=%d row=%d bad=%v visits=%d", workers, active, i, p.bad[i], p.visits[i])
				}
			}
		}
		if e.memory != memory {
			t.Fatal("warm row scratch was replaced")
		}
		if n := testing.AllocsPerRun(1, func() {
			if err := e.RowsWithScratch(p, workers, 1, p.need); err != nil {
				panic(err)
			}
		}); n != 0 {
			t.Fatalf("warm row scratch allocations=%g", n)
		}
		if e.rowJob.op != nil || e.rowJob.scratchOp != nil {
			t.Fatal("idle executor retained row operation")
		}
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
		if memory.Bytes() != 0 {
			t.Fatal("Close retained scratch arena")
		}
		if err := e.RowsWithScratch(p, workers, 1, p.need); err != ErrExecutorClosed {
			t.Fatalf("closed: %v", err)
		}
	}
}

func TestExecutorScratchRowsReuseMatrixArena(t *testing.T) {
	const m, k, n = 65, 256, 128
	a, _, dst, b := benchmarkData(t, m, k, n)
	e, err := NewExecutor(3)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if err := e.Mul(b, dst, n, a, k, m); err != nil {
		t.Fatal(err)
	}
	memory := e.memory
	p := &scratchRowsProbe{executor: e, need: ScratchLen(k), bad: make([]bool, 3), visits: make([]int, 3)}
	if err := e.RowsWithScratch(p, 3, 1, p.need); err != nil {
		t.Fatal(err)
	}
	if e.memory != memory {
		t.Fatal("row operation allocated a second matrix scratch arena")
	}
	for _, bad := range p.bad {
		if bad {
			t.Fatal("worker did not borrow its existing matrix scratch")
		}
	}
	if err := e.Mul(b, dst, n, a, k, m); err != nil {
		t.Fatal(err)
	}
	if e.memory != memory {
		t.Fatal("matrix operation replaced reused scratch")
	}
}

func TestExecutorScratchRowsValidation(t *testing.T) {
	var e Executor
	p := &scratchRowsProbe{executor: &e, bad: make([]bool, 1), visits: make([]int, 1)}
	defer e.Close()
	if err := e.RowsWithScratch(nil, 1, 1, 1); err != ErrNilOperation {
		t.Fatalf("nil: %v", err)
	}
	for _, shape := range [][3]int{{-1, 1, 1}, {1, 0, 1}, {1, 1, -1}} {
		if err := e.RowsWithScratch(p, shape[0], shape[1], shape[2]); err != ErrShape {
			t.Fatalf("shape%v: %v", shape, err)
		}
	}
	if err := e.RowsWithScratch(p, 0, 1, 100); err != nil || e.memory != nil {
		t.Fatal("zero rows allocated")
	}
	if err := e.RowsWithScratch(p, 1, 1, 0); err != nil || p.visits[0] != 1 {
		t.Fatal("zero scratch operation failed")
	}
	var absent *Executor
	if err := absent.RowsWithScratch(p, 1, 1, 1); err != ErrExecutorClosed {
		t.Fatalf("nil executor: %v", err)
	}
}
