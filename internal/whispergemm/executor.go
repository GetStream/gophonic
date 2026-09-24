// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

var (
	ErrWorkers        = errors.New("whispergemm: worker count must be between 1 and 64")
	ErrExecutorClosed = errors.New("whispergemm: executor is closed")
	ErrNilOperation   = errors.New("whispergemm: nil row operation")
)

// Executor reuses a bounded set of workers for matrix and row operations. Each
// worker reads shared inputs and writes disjoint output rows. The
// calling goroutine computes one shard; workers includes that goroutine.
// Small operations run entirely on the caller to avoid worker wakeup costs.
//
// Mul, Rows, and Close serialize on one Executor. Independent executors can share a
// PackedB, provided nobody calls Pack while it is in use. An Executor must not
// be copied after first use, and must be closed to release its worker goroutines.
type Executor struct {
	mu      sync.Mutex
	closed  bool
	workers []executorWorker
	stop    atomic.Bool
	pending atomic.Int32
	stopped sync.WaitGroup
	job     matrixJob
	rowJob  rowJob
}

// Give each worker its own publication counter and cache line. A worker only
// reads the shared job after observing a new generation for its own shard.
type executorWorker struct {
	generation atomic.Uint64
	_          [56]byte
}

type matrixJob struct {
	b                  *PackedB
	a, dst             []float32
	aStride, dstStride int
	m, parts           int
}

// RowOperation performs an operation on rows [start,end). Calls for disjoint
// ranges can run concurrently, so shared inputs must be read-only and each
// call must only write its own rows. ApplyRows must not call the same Executor.
// Pass a preallocated pointer implementation to Rows to avoid interface boxing
// allocations; the operation's own code must also be allocation-free.
type RowOperation interface {
	ApplyRows(start, end int)
}

type rowJob struct {
	op          RowOperation
	rows, parts int
}

// NewExecutor starts workers-1 persistent goroutines. The accepted range is
// 1..64; callers select a budget appropriate for their concurrent sessions.
// The zero value is a valid single-worker Executor, requiring no setup.
func NewExecutor(workers int) (*Executor, error) {
	if workers < 1 || workers > 64 {
		return nil, ErrWorkers
	}
	e := &Executor{}
	if workers == 1 {
		return e, nil
	}
	e.workers = make([]executorWorker, workers-1)
	e.stopped.Add(len(e.workers))
	e.pending.Store(int32(len(e.workers)))
	for i := range e.workers {
		go e.run(i + 1)
	}
	// Construct every worker's reusable sleep timer before returning. It is
	// used only after an idle backoff, never to signal an operation's completion.
	for e.pending.Load() != 0 {
		runtime.Gosched()
	}
	return e, nil
}

func (e *Executor) run(index int) {
	defer e.stopped.Done()
	time.Sleep(50 * time.Microsecond)
	e.pending.Add(-1)
	worker := &e.workers[index-1]
	var previous uint64
	lastWork := time.Now()
	yield := runtime.GOMAXPROCS(0) < len(e.workers)+1
	for idle := 0; !e.stop.Load(); {
		generation := worker.generation.Load()
		if generation == previous {
			// Brief polling serves consecutive operations without parking.
			// Yield periodically so a worker budget above GOMAXPROCS still
			// makes progress, then sleep once the executor is idle. Sleep
			// reuses the timer constructed above and bounds idle CPU usage.
			idle++
			if idle&255 == 0 && time.Since(lastWork) >= 50*time.Microsecond {
				time.Sleep(50 * time.Microsecond)
			} else if yield && idle&63 == 0 {
				runtime.Gosched()
			}
			continue
		}
		previous = generation
		idle = 0
		e.runShard(index)
		// This is the worker's last access to the published operation. The
		// caller observes all worker output before reusing the job storage.
		e.pending.Add(-1)
		lastWork = time.Now()
		yield = runtime.GOMAXPROCS(0) < len(e.workers)+1
	}
}

// Mul has the same shape, aliasing, and numerical contract as PackedB.Mul.
// It allocates no memory after construction, including worker dispatch. The
// packed values are read-only until all shards complete and Mul returns.
func (e *Executor) Mul(b *PackedB, dst []float32, dstStride int, a []float32, aStride int, m int) error {
	if e == nil {
		return ErrExecutorClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return ErrExecutorClosed
	}
	if b == nil {
		return ErrNilMatrix
	}
	if !validInput(a, m, b.k, aStride) || !validMatrix(dst, m, b.n, dstStride) {
		return ErrShape
	}
	parts := e.partitions(m, b.k, b.n)
	if parts == 1 {
		b.mul(dst, dstStride, a, aStride, m)
		return nil
	}
	e.job = matrixJob{b: b, a: a, dst: dst, aStride: aStride, dstStride: dstStride, m: m, parts: parts}
	e.execute(parts)
	// Release the caller's buffers and matrix while idle. The atomic completion
	// barrier ensures workers no longer read the job before this reset.
	e.job = matrixJob{}
	return nil
}

// Rows applies op to every row in [0,rows), using the same persistent workers
// as Mul. rows must be nonnegative and minRows must be positive. minRows limits
// scheduling to at least that many rows per worker; smaller inputs run on the
// caller. Zero rows does not call ApplyRows. A nil interface returns
// ErrNilOperation; callers must not pass a typed nil pointer. Rows does not allocate after construction when op
// is a reused pointer whose ApplyRows implementation does not allocate.
func (e *Executor) Rows(op RowOperation, rows, minRows int) error {
	if e == nil {
		return ErrExecutorClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return ErrExecutorClosed
	}
	if op == nil {
		return ErrNilOperation
	}
	if rows < 0 || minRows < 1 {
		return ErrShape
	}
	if rows == 0 {
		return nil
	}
	parts := max(1, min(len(e.workers)+1, rows/minRows))
	if parts == 1 {
		op.ApplyRows(0, rows)
		return nil
	}
	e.rowJob = rowJob{op: op, rows: rows, parts: parts}
	e.execute(parts)
	e.rowJob = rowJob{}
	return nil
}

func (e *Executor) execute(parts int) {
	e.pending.Store(int32(parts - 1))
	for i := 0; i < parts-1; i++ {
		e.workers[i].generation.Add(1)
	}
	e.runShard(0)
	// Balanced shards normally finish together. Poll briefly before yielding
	// to runnable workers; yielding is required when GOMAXPROCS is below the
	// worker budget. Avoiding a semaphore wait keeps this hot barrier free of
	// runtime waiter allocations even when the caller migrates between Ps.
	yield := runtime.GOMAXPROCS(0) < parts
	for spins := 0; e.pending.Load() != 0; spins++ {
		if yield && spins&63 == 63 {
			runtime.Gosched()
		}
	}
}

func (e *Executor) runShard(index int) {
	if j := &e.rowJob; j.op != nil {
		perPart, remainder := j.rows/j.parts, j.rows%j.parts
		start := index*perPart + min(index, remainder)
		end := start + perPart
		if index < remainder {
			end++
		}
		j.op.ApplyRows(start, end)
		return
	}
	e.mulShard(index)
}

func (e *Executor) partitions(m, k, n int) int {
	if m < 32 || k == 0 || n == 0 || len(e.workers) == 0 {
		return 1
	}
	// At least 16 rows and roughly 1M multiply-adds per shard. Divide twice
	// using ceiling division to avoid overflowing m*k*n for hostile shapes.
	const minWork = 1 << 20
	minRows := (minWork-1)/n + 1
	minRows = (minRows-1)/k + 1
	minRows = max(16, (minRows+3)/4*4)
	return max(1, min(len(e.workers)+1, m/minRows))
}

func (e *Executor) mulShard(index int) {
	j := &e.job
	// Keep every boundary on a four-row tile so the numerical reduction
	// order is identical to a single PackedB.Mul, including the final tail.
	blocks := j.m/4 + min(j.m%4, 1)
	perPart, remainder := blocks/j.parts, blocks%j.parts
	first := index*perPart + min(index, remainder)
	count := perPart
	if index < remainder {
		count++
	}
	start, end := first*4, min((first+count)*4, j.m)
	j.b.mul(j.dst[start*j.dstStride:], j.dstStride, j.a[start*j.aStride:], j.aStride, end-start)
}

// Close waits for any active operation and stops the workers. Repeated
// Close calls, including on a nil receiver, succeed. Subsequent Mul and Rows calls
// return ErrExecutorClosed.
// Workers returns the configured worker limit, including the caller.
func (e *Executor) Workers() int {
	if e == nil {
		return 1
	}
	return len(e.workers) + 1
}

func (e *Executor) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	e.stop.Store(true)
	e.stopped.Wait()
	return nil
}
