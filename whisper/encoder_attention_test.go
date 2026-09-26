// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"fmt"
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

func TestPackedAudioAttentionMatchesScalar(t *testing.T) {
	for _, shape := range []struct{ rows, state, heads int }{
		{1, 8, 2}, {2, 2, 1}, {7, 15, 3}, {17, 32, 4}, {65, 384, 6},
	} {
		t.Run(fmt.Sprintf("%dx%d/%d", shape.rows, shape.state, shape.heads), func(t *testing.T) {
			n := shape.rows * shape.state
			q, k, v := make([]float32, n), make([]float32, n), make([]float32, n)
			for i := 0; i < n; i++ {
				q[i] = float32(math.Sin(float64(i)*0.37)) * 3
				k[i] = float32(math.Cos(float64(i)*0.19)) * 2
				v[i] = float32(math.Sin(float64(i) * 0.53))
			}
			want := append([]float32(nil), q...)
			refK := append([]float32(nil), k...)
			multiHeadAttention(want, refK, v, want, make([]float32, shape.rows), shape.rows, shape.state, shape.heads)
			for _, workers := range []int{1, 4} {
				attention, err := newAudioAttention(shape.rows, shape.state, shape.heads, workers)
				if err != nil {
					t.Fatal(err)
				}
				e, err := whispergemm.NewExecutor(workers)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { e.Close(); attention.close() })
				got := append([]float32(nil), q...)
				gotK := append([]float32(nil), k...)
				if err := attention.run(got, gotK, v, got, e); err != nil {
					t.Fatal(err)
				}
				for i, value := range got {
					if delta := math.Abs(float64(value - want[i])); math.IsNaN(float64(value)) || delta > 3e-6 {
						t.Fatalf("workers=%d output[%d]=%.9g, want %.9g", workers, i, value, want[i])
					}
				}
				allocs := testing.AllocsPerRun(3, func() {
					copy(got, q)
					copy(gotK, k)
					if err := attention.run(got, gotK, v, got, e); err != nil {
						panic(err)
					}
				})
				if allocs != 0 {
					t.Fatalf("workers=%d warm attention allocated %g objects", workers, allocs)
				}
			}
		})
	}
}

func TestSoftmaxRowsStableReference(t *testing.T) {
	values := []float32{1000, 1001, 1002, -1000, -1001, -1002}
	softmaxRows(values, 2, 3)
	denominator := 1 + math.Exp(-1) + math.Exp(-2)
	want := []float64{math.Exp(-2) / denominator, math.Exp(-1) / denominator, 1 / denominator,
		1 / denominator, math.Exp(-1) / denominator, math.Exp(-2) / denominator}
	for i, value := range values {
		if math.Abs(float64(value)-want[i]) > 1e-7 {
			t.Fatalf("softmax[%d]=%.9g, want %.9g", i, value, want[i])
		}
	}
}

func BenchmarkAudioSoftmax(b *testing.B) {
	const rows, columns = attentionTileRows, AudioFrames
	src, values := make([]float32, rows*columns), make([]float32, rows*columns)
	for i := range src {
		src[i] = float32(math.Sin(float64(i)*0.07)*8 + math.Cos(float64(i)*0.013)*4)
	}
	b.ReportAllocs()
	for b.Loop() {
		copy(values, src)
		softmaxRows(values, rows, columns)
	}
	encoderBenchmarkSink = values[len(values)-1]
}
