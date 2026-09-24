// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"fmt"
	"sync"
	"testing"

	"github.com/GetStream/gophonic/internal/q8gemm"
)

type projOnly struct {
	w   *q8gemm.Weights
	ws  *q8gemm.Workspace
	dst []float32
	n   int
}

func (o *projOnly) ApplyRows(_, start, end int) {
	if err := q8gemm.MulPanels(o.dst, o.n, o.ws, o.w, start, end, nil); err != nil {
		panic(err)
	}
}

// BenchmarkProjectionStreaming streams 40 distinct 12288x4096 matrices
// (2 GB) through 16-row tiles to measure DRAM-bound projection throughput.
func BenchmarkProjectionStreaming(b *testing.B) {
	if !q8gemm.Available() {
		b.Skip("no SME")
	}
	const k, n, mats = 4096, 12288, 40
	q := make([]int8, k*n)
	for i := range q {
		q[i] = int8(i*7%255 - 127)
	}
	scales := make([]float32, n)
	for i := range scales {
		scales[i] = 0.01
	}
	weights := make([]*q8gemm.Weights, mats)
	for i := range weights {
		weights[i], _ = q8gemm.NewWeights(k, n)
		if err := weights[i].Pack(q, scales); err != nil {
			b.Fatal(err)
		}
	}
	ws, _ := q8gemm.NewWorkspace(k)
	x := make([]float32, 16*k)
	for i := range x {
		x[i] = float32(i%13) * 0.1
	}
	if err := ws.Pack(x, 16, k); err != nil {
		b.Fatal(err)
	}
	dst := make([]float32, 16*n)
	for _, threads := range []int{1, 8, 10, 12} {
		b.Run(fmt.Sprintf("pool-%d", threads), func(b *testing.B) {
			pool := newWorkerPool(threads)
			defer pool.close()
			op := &projOnly{ws: ws, dst: dst, n: n}
			for b.Loop() {
				for _, w := range weights {
					op.w = w
					pool.run(op, w.Panels(), 1)
				}
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/1e6/float64(mats*k*n)*6.95e9, "ms/7GB-pass")
		})
		b.Run(fmt.Sprintf("static-%d", threads), func(b *testing.B) {
			for b.Loop() {
				var wg sync.WaitGroup
				for t := range threads {
					wg.Add(1)
					go func() {
						defer wg.Done()
						for _, w := range weights {
							p := w.Panels()
							if err := q8gemm.MulPanels(dst, n, ws, w, p*t/threads, p*(t+1)/threads, nil); err != nil {
								panic(err)
							}
						}
					}()
				}
				wg.Wait()
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/1e6/float64(mats*k*n)*6.95e9, "ms/7GB-pass")
		})
	}
}
