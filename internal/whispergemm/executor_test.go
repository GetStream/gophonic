// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"math"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestExecutorMatchesSingleThread(t *testing.T) {
	for _, workers := range []int{1, 2, 3, 8} {
		e, err := NewExecutor(workers)
		if err != nil {
			t.Fatal(err)
		}
		for _, shape := range [][3]int{
			{0, 0, 0}, {5, 0, 9}, {9, 5, 0}, {1, 384, 19},
			{31, 65, 19}, {32, 384, 384}, {65, 256, 128},
			{99, 511, 73}, {127, 385, 193}, {129, 256, 256},
		} {
			m, k, n := shape[0], shape[1], shape[2]
			as, ds := k+3, n+5
			a, w := make([]float32, m*as), make([]float32, k*n)
			state := uint32(0x9876)
			for i := range a {
				a[i] = nextValue(&state)
			}
			for i := range w {
				w[i] = nextValue(&state)
			}
			b, _ := NewPackedB(k, n)
			if err := b.Pack(w, k, true); err != nil {
				t.Fatal(err)
			}
			want, got := make([]float32, m*ds), make([]float32, m*ds)
			for i := range want {
				want[i], got[i] = -37, -37
			}
			if err := b.Mul(want, ds, a, as, m); err != nil {
				t.Fatal(err)
			}
			for range 3 {
				if err := e.Mul(b, got, ds, a, as, m); err != nil {
					t.Fatal(err)
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("workers=%d shape=%v output[%d]=%g want %g", workers, shape, i, got[i], want[i])
					}
				}
			}
		}
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestExecutorValidationAndLifetime(t *testing.T) {
	for _, workers := range []int{-1, 0, 65, 1 << 30} {
		if _, err := NewExecutor(workers); err != ErrWorkers {
			t.Fatalf("workers=%d: %v", workers, err)
		}
	}
	b, _ := NewPackedB(1, 1)
	if err := b.Pack([]float32{3}, 1, false); err != nil {
		t.Fatal(err)
	}
	var e Executor
	dst := make([]float32, 1)
	if err := e.Mul(b, dst, 1, []float32{7}, 1, 1); err != nil || dst[0] != 21 {
		t.Fatalf("zero executor output=%v error=%v", dst, err)
	}
	if err := e.Mul(nil, nil, 0, nil, 0, 0); err != ErrNilMatrix {
		t.Fatalf("nil matrix: %v", err)
	}
	if err := e.Mul(b, nil, 1, nil, 1, 1); err != ErrShape {
		t.Fatalf("invalid matrix: %v", err)
	}
	for range 2 {
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.Mul(b, dst, 1, []float32{7}, 1, 1); err != ErrExecutorClosed {
		t.Fatalf("closed executor: %v", err)
	}
	var nilExecutor *Executor
	if err := nilExecutor.Close(); err != nil {
		t.Fatal(err)
	}
	if err := nilExecutor.Mul(b, dst, 1, []float32{7}, 1, 1); err != ErrExecutorClosed {
		t.Fatalf("nil executor: %v", err)
	}
}

func TestExecutorConcurrentCallsAndZeroAllocations(t *testing.T) {
	const m, k, n = 129, 256, 256
	a, _, dst, b := benchmarkData(t, m, k, n)
	e, err := NewExecutor(4)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if e.partitions(m, k, n) != 4 {
		t.Fatal("test must exercise all worker shards")
	}
	rowDst := make([]float32, len(dst))
	op := &copyRowsOperation{dst: rowDst, src: dst}
	if allocations := testing.AllocsPerRun(5, func() {
		if err := e.Mul(b, dst, n, a, k, m); err != nil {
			panic(err)
		}
		if err := e.Rows(op, len(dst), 256); err != nil {
			panic(err)
		}
	}); allocations != 0 {
		t.Fatalf("Executor.Mul + Rows allocated %g objects", allocations)
	}
	if e.job.b != nil || e.job.a != nil || e.job.dst != nil || e.rowJob.op != nil {
		t.Fatal("idle executor retained caller buffers")
	}
	var calls sync.WaitGroup
	for range 4 {
		calls.Go(func() {
			got := make([]float32, len(dst))
			op := &copyRowsOperation{dst: got, src: dst}
			for range 3 {
				if err := e.Mul(b, got, n, a, k, m); err != nil {
					t.Error(err)
					return
				}
				for i := range got {
					if got[i] != dst[i] {
						t.Errorf("concurrent output[%d]=%g want %g", i, got[i], dst[i])
						return
					}
				}
				clear(got)
				if err := e.Rows(op, len(dst), 256); err != nil {
					t.Error(err)
					return
				}
				for i := range got {
					if got[i] != dst[i] {
						t.Errorf("concurrent Rows output[%d]=%g want %g", i, got[i], dst[i])
						return
					}
				}
			}
		})
	}
	calls.Wait()
}

func TestExecutorOversubscribedAndIdleResume(t *testing.T) {
	previous := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(previous)
	for _, procs := range []int{1, 2} {
		runtime.GOMAXPROCS(procs)
		e, err := NewExecutor(8)
		if err != nil {
			t.Fatal(err)
		}
		src, dst := make([]float32, 8192), make([]float32, 8192)
		op := &copyRowsOperation{src: src, dst: dst}
		for pass := range 3 {
			// Let all workers enter their idle backoff before republishing a
			// job. Varying the active shard count catches stale generations.
			time.Sleep(2 * time.Millisecond)
			for _, rows := range []int{8192, 3, 67, 1, 8, 17} {
				for i := range src {
					src[i] = float32(i + 1 + pass*8192)
				}
				clear(dst)
				if err := e.Rows(op, rows, 1); err != nil {
					t.Fatal(err)
				}
				for i, value := range dst {
					want := float32(0)
					if i < rows {
						want = src[i]
					}
					if value != want {
						t.Fatalf("GOMAXPROCS=%d pass=%d rows=%d output[%d]=%g want %g", procs, pass, rows, i, value, want)
					}
				}
			}
		}
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestExecutorCloseWaitsForActiveRows(t *testing.T) {
	e, err := NewExecutor(4)
	if err != nil {
		t.Fatal(err)
	}
	release := make(chan struct{})
	op := &blockingRowsOperation{visits: make([]int, 4), arrived: make(chan int, 4), release: release}
	operationDone := make(chan error, 1)
	go func() { operationDone <- e.Rows(op, 4, 1) }()
	for range 4 {
		<-op.arrived
	}
	closeStarted, closeDone := make(chan struct{}), make(chan error, 1)
	go func() {
		close(closeStarted)
		closeDone <- e.Close()
	}()
	<-closeStarted
	premature := false
	select {
	case <-closeDone:
		premature = true
	case <-time.After(time.Millisecond):
	}
	close(release)
	if err := <-operationDone; err != nil {
		t.Fatal(err)
	}
	if premature {
		t.Fatal("Close returned before the active row operation completed")
	}
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not stop idle workers")
	}
}

// AllocsPerRun pins GOMAXPROCS to one, so it cannot catch runtime waiter
// allocation caused by a multi-P worker pool after a garbage collection.
func TestExecutorRowsDoNotAllocateAfterGC(t *testing.T) {
	previous := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(previous)
	previousProfileRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() { runtime.MemProfileRate = previousProfileRate }()
	e, err := NewExecutor(8)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	src, dst := make([]float32, 8192), make([]float32, 8192)
	for i := range src {
		src[i] = float32(i)
	}
	op := &copyRowsOperation{dst: dst, src: src}
	for range 16 {
		if err := e.Rows(op, len(src), 64); err != nil {
			t.Fatal(err)
		}
	}
	runtime.GC()
	before := executorProfileAllocations()
	for range 128 {
		if err := e.Rows(op, len(src), 64); err != nil {
			t.Fatal(err)
		}
	}
	// A collection publishes the allocation profile. Check allocations whose
	// stack contains this executor, excluding independent runtime tasks such
	// as the scavenger's timer, which can run just after a forced GC.
	runtime.GC()
	after := executorProfileAllocations()
	if after != before {
		t.Fatalf("multi-P row operations after GC allocated %d objects", after-before)
	}
	for i, value := range dst {
		if value != src[i] {
			t.Fatalf("output[%d]=%g, want %g", i, value, src[i])
		}
	}
}

func executorProfileAllocations() int64 {
	n, _ := runtime.MemProfile(nil, true)
	for {
		records := make([]runtime.MemProfileRecord, n+32)
		count, ok := runtime.MemProfile(records, true)
		if !ok {
			n = count
			continue
		}
		var allocations int64
		for _, record := range records[:count] {
			for _, pc := range record.Stack() {
				function := runtime.FuncForPC(pc)
				if function != nil && strings.Contains(function.Name(), "whispergemm.(*Executor).") {
					allocations += record.AllocObjects
					break
				}
			}
		}
		return allocations
	}
}

type copyRowsOperation struct {
	dst, src []float32
}

func (op *copyRowsOperation) ApplyRows(start, end int) {
	copy(op.dst[start:end], op.src[start:end])
}

type blockingRowsOperation struct {
	visits  []int
	arrived chan int
	release <-chan struct{}
}

func (op *blockingRowsOperation) ApplyRows(start, end int) {
	for i := start; i < end; i++ {
		op.visits[i]++
	}
	op.arrived <- start
	<-op.release
}

func TestExecutorRowsPartitioning(t *testing.T) {
	for _, rows := range []int{4, 67} {
		e, err := NewExecutor(4)
		if err != nil {
			t.Fatal(err)
		}
		release := make(chan struct{})
		op := &blockingRowsOperation{visits: make([]int, rows), arrived: make(chan int, 4), release: release}
		completed := make(chan error, 1)
		go func() { completed <- e.Rows(op, rows, 1) }()
		// Every shard must enter before any is released. This proves actual
		// concurrent worker execution, including one-row-per-worker jobs.
		timer := time.NewTimer(5 * time.Second)
		starts := make(map[int]bool)
		for range 4 {
			select {
			case start := <-op.arrived:
				starts[start] = true
			case <-timer.C:
				close(release)
				e.Close()
				t.Fatal("row shards did not execute concurrently")
			}
		}
		timer.Stop()
		close(release)
		if err := <-completed; err != nil {
			t.Fatal(err)
		}
		e.Close()
		if len(starts) != 4 {
			t.Fatalf("worker ranges overlap: %v", starts)
		}
		for row, visits := range op.visits {
			if visits != 1 {
				t.Fatalf("row %d visited %d times", row, visits)
			}
		}
	}
}

func TestExecutorRowsValidationAndSmallInputs(t *testing.T) {
	e, err := NewExecutor(4)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	dst, src := make([]float32, 19), make([]float32, 19)
	for i := range src {
		src[i] = float32(i + 1)
	}
	op := &copyRowsOperation{dst: dst, src: src}
	for _, shape := range [][2]int{{0, 1}, {1, 1}, {7, 8}, {19, 4}, {19, 20}} {
		clear(dst)
		if err := e.Rows(op, shape[0], shape[1]); err != nil {
			t.Fatal(err)
		}
		for i, got := range dst {
			var want float32
			if i < shape[0] {
				want = src[i]
			}
			if got != want {
				t.Fatalf("rows=%d minRows=%d row=%d: got %g want %g", shape[0], shape[1], i, got, want)
			}
		}
	}
	if err := e.Rows(nil, 0, 1); err != ErrNilOperation {
		t.Fatalf("nil operation: %v", err)
	}
	for _, shape := range [][2]int{{-1, 1}, {1, 0}, {1, -1}} {
		if err := e.Rows(op, shape[0], shape[1]); err != ErrShape {
			t.Fatalf("Rows(%d,%d): %v", shape[0], shape[1], err)
		}
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if err := e.Rows(op, 1, 1); err != ErrExecutorClosed {
		t.Fatalf("closed executor: %v", err)
	}
	var nilExecutor *Executor
	if err := nilExecutor.Rows(op, 1, 1); err != ErrExecutorClosed {
		t.Fatalf("nil executor: %v", err)
	}
}

// Each executor fits GOMAXPROCS by itself; together they do not. The caller
// must release its P while a shard sleeps, including after GOMAXPROCS changes.
func TestPrivateExecutorsConcurrentProgress(t *testing.T) {
	previous := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(previous)
	const count = 4
	var executors [count]*Executor
	var ops [count]delayedPrivateRows
	for i := range executors {
		var err error
		executors[i], err = NewExecutor(2)
		if err != nil {
			t.Fatal(err)
		}
		defer executors[i].Close()
	}
	for _, procs := range []int{2, 1, 4} {
		runtime.GOMAXPROCS(procs)
		var done sync.WaitGroup
		for i, e := range executors {
			done.Go(func() {
				for range 12 {
					if err := e.Rows(&ops[i], 128, 1); err != nil {
						t.Error(err)
						return
					}
				}
			})
		}
		done.Wait()
	}
	for lane := range ops {
		for i, v := range ops[lane].values {
			if v != 36*(i+1) {
				t.Fatalf("lane%d row%d=%d", lane, i, v)
			}
		}
	}
}

type delayedPrivateRows struct{ values [128]int }

func (o *delayedPrivateRows) ApplyRows(start, end int) {
	if start > 0 {
		time.Sleep(50 * time.Microsecond)
	}
	for i := start; i < end; i++ {
		o.values[i] += i + 1
	}
}
