// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3asr

import (
	"bytes"
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/GetStream/gophonic/speech"
)

// TestGPUConcurrentTranscribers compares concurrent public calls against
// serial numeric fingerprints. Only the model weights are shared; every
// transcriber owns its encoder, decoder scratch, output, and prefix cache.
func TestGPUConcurrentTranscribers(t *testing.T) {
	m := loadModel(t, "gpu-q8")
	var tr [2]*Transcriber
	var pcm [2][]float32
	var out [2]speech.Transcript
	var want [2][32]byte
	for i, clip := range []string{"jfk", "zh"} {
		var err error
		tr[i], err = NewTranscriber(m, LaneOptions{Threads: 1})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { tr[i].Close() })
		pcm[i] = clipPCM(t, clip)
		if err := tr[i].Transcribe(context.Background(), pcm[i], speech.Options{}, &out[i]); err != nil {
			t.Fatal(err)
		}
		want[i] = continuationFingerprint(tr[i], &out[i])
	}
	for round := range 3 {
		var wg sync.WaitGroup
		var errs [2]error
		start := make(chan struct{})
		for i := range tr {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				errs[i] = tr[i].Transcribe(context.Background(), pcm[i], speech.Options{}, &out[i])
			}()
		}
		close(start)
		wg.Wait()
		for i := range tr {
			if errs[i] != nil {
				t.Fatal(errs[i])
			}
			if got := continuationFingerprint(tr[i], &out[i]); got != want[i] {
				t.Fatalf("round %d lane %d changed numeric state or transcript", round, i)
			}
		}
	}
}

// BenchmarkGPUConcurrentTranscribe measures complete warm PCM-to-text calls
// through independent persistent workers. One operation completes one call
// per lane; calls/s reports aggregate throughput, not single-call latency.
func BenchmarkGPUConcurrentTranscribe(b *testing.B) {
	m := loadModel(b, "gpu-q8")
	pcm := clipPCM(b, "jfk")
	for _, count := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("lanes=%d", count), func(b *testing.B) {
			inputs := make([][]float32, count)
			for i := range inputs {
				inputs[i] = pcm
			}
			calls := newGPUConcurrentCalls(b, m, inputs)
			b.ReportAllocs()
			for b.Loop() {
				if err := calls.run(); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.N*count)/b.Elapsed().Seconds(), "calls/s")
			b.ReportMetric(float64(count), "calls/op")
		})
	}
}

// gpuConcurrentCalls is the same persistent independent-worker baseline for
// the standalone and interleaved benchmarks. All transcript checks are inside
// run, and both serial and concurrent warmup are outside measurement.
type gpuConcurrentCalls struct {
	jobs    []chan struct{}
	results chan error
	stopped sync.WaitGroup
	want    [][]byte
}

func newGPUConcurrentCalls(tb testing.TB, m *Model, inputs [][]float32) *gpuConcurrentCalls {
	tb.Helper()
	r := &gpuConcurrentCalls{jobs: make([]chan struct{}, len(inputs)), results: make(chan error, len(inputs)), want: make([][]byte, len(inputs))}
	tb.Cleanup(func() {
		for _, jobs := range r.jobs {
			if jobs != nil {
				close(jobs)
			}
		}
		r.stopped.Wait()
	})
	for i, pcm := range inputs {
		tr, err := NewTranscriber(m, LaneOptions{Threads: 1})
		if err != nil {
			tb.Fatal(err)
		}
		var dst speech.Transcript
		if err := tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
			tr.Close()
			tb.Fatal(err)
		}
		want := slices.Clone(dst.Text)
		r.want[i] = want
		r.jobs[i] = make(chan struct{}, 1)
		r.stopped.Add(1)
		go func(jobs <-chan struct{}) {
			defer r.stopped.Done()
			defer tr.Close()
			for range jobs {
				err := tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst)
				if err == nil && !bytes.Equal(dst.Text, want) {
					err = fmt.Errorf("concurrent transcript changed")
				}
				r.results <- err
			}
		}(r.jobs[i])
	}
	for range 2 {
		if err := r.run(); err != nil {
			tb.Fatal(err)
		}
	}
	return r
}

func (r *gpuConcurrentCalls) run() error {
	for _, jobs := range r.jobs {
		jobs <- struct{}{}
	}
	var first error
	for range r.jobs {
		if err := <-r.results; first == nil {
			first = err
		}
	}
	return first
}
