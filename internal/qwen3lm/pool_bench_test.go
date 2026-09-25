// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"fmt"
	"testing"
)

type noopOp struct{ sink [64]int }

func (o *noopOp) ApplyRows(_, start, end int) { o.sink[start%64] += end - start }

func BenchmarkPoolDispatch(b *testing.B) {
	for _, workers := range []int{1, 4, 8, 10, 12} {
		b.Run(fmt.Sprint(workers), func(b *testing.B) {
			p := newWorkerPool(workers)
			defer p.close()
			op := &noopOp{}
			for b.Loop() {
				p.run(op, 64, 1)
			}
		})
	}
}
