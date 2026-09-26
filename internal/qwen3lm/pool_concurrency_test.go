// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"runtime"
	"sync"
	"testing"
	"time"
)

type privatePoolRows struct{ values [128]uint64 }

func (o *privatePoolRows) ApplyRows(worker, start, end int) {
	// A delayed shard exercises a caller waiting while other private pools
	// occupy the available Ps. Different pools never touch the same values.
	if worker != 0 {
		time.Sleep(50 * time.Microsecond)
	}
	for i := start; i < end; i++ {
		o.values[i] = o.values[i]*3 + uint64(i+1)
	}
}

func TestPrivatePoolsConcurrentProgress(t *testing.T) {
	previous := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(previous)
	const count = 4
	var pools [count]*workerPool
	var rows [count]privatePoolRows
	for i := range pools {
		pools[i] = newWorkerPool(2)
		defer pools[i].close()
	}
	for _, procs := range []int{2, 1, 4} {
		runtime.GOMAXPROCS(procs)
		var done sync.WaitGroup
		for i, p := range pools {
			done.Go(func() {
				p.hold()
				defer p.release()
				for range 12 {
					p.run(&rows[i], 128, 8)
					p.runN(&rows[i], 128, 8, 1)
					p.runEach(&rows[i], 2)
				}
			})
		}
		done.Wait()
	}
	for lane := range rows {
		for i, got := range rows[lane].values {
			var want uint64
			for range 36 {
				want = want*3 + uint64(i+1)
				want = want*3 + uint64(i+1)
				if i < 2 {
					want = want*3 + uint64(i+1)
				}
			}
			if got != want {
				t.Fatalf("lane%d row%d=%d want %d", lane, i, got, want)
			}
		}
		if pools[lane].op != nil {
			t.Fatal("idle pool retained operation")
		}
	}
}
