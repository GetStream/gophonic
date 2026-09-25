// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"fmt"
	"testing"

	"github.com/GetStream/gophonic/internal/q8gemm"
)

type projI8 struct {
	w     *q8gemm.WeightsI8
	tiles []*q8gemm.WorkspaceI8
	dst   []float32
	n     int
}

func (o *projI8) ApplyRows(_, start, end int) {
	for p := start; p < end; p++ {
		for t, ws := range o.tiles {
			if err := q8gemm.MulPanelsI8(o.dst[t*16*o.n:], o.n, ws, o.w, p, p+1); err != nil {
				panic(err)
			}
		}
	}
}

// BenchmarkProjectionStreamingInt8 streams 40 distinct 12288x4096 int8
// matrices (2 GB) through 1 and 4 activation tiles, reporting the time a
// full Qwen3-8B pass (6.95G weights) would take per 16-row tile.
func BenchmarkProjectionStreamingInt8(b *testing.B) {
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
	weights := make([]*q8gemm.WeightsI8, mats)
	for i := range weights {
		weights[i], _ = q8gemm.NewWeightsI8(k, n)
		if err := weights[i].Pack(q, scales); err != nil {
			b.Fatal(err)
		}
	}
	x := make([]float32, 16*k)
	for i := range x {
		x[i] = float32(i%13) - 6
	}
	for _, tiles := range []int{1, 4} {
		for _, threads := range []int{4, 8} {
			b.Run(fmt.Sprintf("tiles-%d/threads-%d", tiles, threads), func(b *testing.B) {
				op := &projI8{n: n, dst: make([]float32, tiles*16*n)}
				for range tiles {
					ws, _ := q8gemm.NewWorkspaceI8(k)
					_ = ws.Prepare(16, k)
					for r := range 16 {
						ws.SetRowScale(r, q8gemm.MaxAbs(x[r*k:(r+1)*k]))
					}
					_ = ws.PackRange(x, k, 0, k)
					op.tiles = append(op.tiles, ws)
				}
				pool := newWorkerPool(threads)
				defer pool.close()
				for b.Loop() {
					for _, w := range weights {
						op.w = w
						pool.run(op, w.Panels(), 1)
					}
				}
				perPass := b.Elapsed().Seconds() / float64(b.N) / float64(mats*k*n) * 6.95e9
				b.ReportMetric(perPass*1e3/float64(tiles), "ms/tile")
			})
		}
	}
}
