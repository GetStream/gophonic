// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"runtime"
	"strconv"
	"testing"
)

type tinyBarrierWorker struct {
	start  chan struct{}
	done   chan struct{}
	exited chan struct{}
}

func newTinyBarrierWorkers(count int) []tinyBarrierWorker {
	workers := make([]tinyBarrierWorker, count)
	for i := range workers {
		worker := &workers[i]
		worker.start = make(chan struct{}, 1)
		worker.done = make(chan struct{}, 1)
		worker.exited = make(chan struct{})
		go func() {
			defer close(worker.exited)
			for range worker.start {
				worker.done <- struct{}{}
			}
		}()
	}
	return workers
}

func BenchmarkTinyWorkerBarrier(b *testing.B) {
	helperLimit := runtime.GOMAXPROCS(0) - 1
	if helperLimit > tinyMaxWorkers {
		helperLimit = tinyMaxWorkers
	}
	for helpers := 0; helpers <= helperLimit; helpers++ {
		b.Run("helpers="+strconv.Itoa(helpers), func(b *testing.B) {
			workers := newTinyBarrierWorkers(helpers)
			defer func() {
				for i := range workers {
					close(workers[i].start)
				}
				for i := range workers {
					<-workers[i].exited
				}
			}()
			barrier := func(active int) {
				for i := 0; i < active; i++ {
					workers[i].start <- struct{}{}
				}
				for i := 0; i < active; i++ {
					<-workers[i].done
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// The Tiny graph has four dense convolution stages and one
				// independent reverse-GRU direction dispatched to a helper.
				for stage := 0; stage < 4; stage++ {
					barrier(helpers)
				}
				if helpers > 0 {
					barrier(1)
				}
			}
		})
	}
}
