// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// rangeOp is one parallel operation over items [0,n). ApplyRows receives
// disjoint ranges, possibly concurrently, with the index of the participant
// running it (0 is the caller, then 1..workers-1); it must write only its own
// items and may use per-participant scratch.
type rangeOp interface {
	ApplyRows(worker, start, end int)
}

// workerPool runs one rangeOp at a time on persistent goroutines plus the
// caller. A forward pass issues hundreds of short operations back to back, so
// idle workers spin for poolSpin before parking; parked workers are woken by a
// channel token. Items are claimed in small fixed grains (see claim), so a descheduled
// or slower core does not hold a fixed share of the work. Dispatch allocates
// nothing after construction.
type workerPool struct {
	mu      sync.Mutex // serializes run and close
	workers []poolWorker
	yield   atomic.Bool  // spinning must yield: more participants than GOMAXPROCS
	active  atomic.Int32 // holds from forward passes; workers never park while positive
	stop    atomic.Bool
	stopped sync.WaitGroup

	// Published job; read by workers only after observing a new generation.
	op      rangeOp
	items   int
	grain   int
	next    atomic.Int64
	pending atomic.Int32 // workers that have not finished the current job
}

type poolWorker struct {
	generation atomic.Uint64
	parked     atomic.Bool
	wake       chan struct{}
	_          [40]byte
}

const poolSpin = time.Millisecond

func newWorkerPool(workers int) *workerPool {
	p := &workerPool{workers: make([]poolWorker, max(0, workers-1))}
	for i := range p.workers {
		p.workers[i].wake = make(chan struct{}, 1)
	}
	p.stopped.Add(len(p.workers))
	for i := range p.workers {
		go p.loop(&p.workers[i], i+1)
	}
	return p
}

// size reports the number of participants, including the caller.
func (p *workerPool) size() int {
	if p == nil {
		return 1
	}
	return len(p.workers) + 1
}

func (p *workerPool) loop(w *poolWorker, index int) {
	defer p.stopped.Done()
	var seen uint64
	for {
		spinStart := time.Now()
		for spins := 0; w.generation.Load() == seen; spins++ {
			if p.stop.Load() {
				return
			}
			if spins&1023 == 1023 {
				// Inside a forward pass the next dispatch is always near, so
				// only an idle pool (no hold) parks after poolSpin.
				if p.active.Load() > 0 || time.Since(spinStart) < poolSpin {
					if p.yield.Load() {
						runtime.Gosched()
					}
					continue
				}
				// Park. Re-check after publishing parked so a concurrent
				// dispatch either sees parked (and sends a token) or this
				// worker sees its new generation.
				w.parked.Store(true)
				if w.generation.Load() != seen || p.stop.Load() {
					if !w.parked.CompareAndSwap(true, false) {
						<-w.wake // the dispatcher already sent a token
					}
					continue
				}
				<-w.wake
				spinStart = time.Now()
			}
		}
		seen = w.generation.Load()
		if p.stop.Load() {
			return
		}
		p.claim(index)
		p.pending.Add(-1)
	}
}

// claim takes fixed grains from a shared counter. Per-thread SME throughput
// varies with how many threads share a cluster's matrix unit, so any larger
// pre-sized share can finish late and stall the job barrier; with one grain
// per claim the barrier waits for at most one grain.
func (p *workerPool) claim(worker int) {
	grain := int64(p.grain)
	items := int64(p.items)
	for {
		start := p.next.Add(grain) - grain
		if start >= items {
			return
		}
		p.op.ApplyRows(worker, int(start), int(min(start+grain, items)))
	}
}

// run applies op to [0,items) in claims of grain items (the last may be
// shorter).
func (p *workerPool) run(op rangeOp, items, grain int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.workers) == 0 || items <= grain || p.stop.Load() {
		op.ApplyRows(0, 0, items)
		return
	}
	// GOMAXPROCS may change at run time (testing.AllocsPerRun sets it to 1).
	// Spinning without yielding is only safe when every participant has a P.
	yield := len(p.workers)+1 > runtime.GOMAXPROCS(0)
	p.yield.Store(yield)
	p.op, p.items, p.grain = op, items, grain
	p.next.Store(0)
	p.pending.Store(int32(len(p.workers)))
	for i := range p.workers {
		w := &p.workers[i]
		w.generation.Add(1)
		if w.parked.CompareAndSwap(true, false) {
			w.wake <- struct{}{}
		}
	}
	p.claim(0)
	for spins := 0; p.pending.Load() != 0; spins++ {
		if yield && spins&255 == 255 {
			runtime.Gosched()
		}
	}
	p.op = nil
}

// hold keeps workers spinning until the matching release, so the many short
// dispatches of one forward pass never pay a parked worker's wake-up.
func (p *workerPool) hold() {
	if p != nil {
		p.active.Add(1)
	}
}

func (p *workerPool) release() {
	if p != nil {
		p.active.Add(-1)
	}
}

func (p *workerPool) close() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.stop.Swap(true) {
		return
	}
	for i := range p.workers {
		w := &p.workers[i]
		if w.parked.CompareAndSwap(true, false) {
			w.wake <- struct{}{}
		}
	}
	p.stopped.Wait()
}
