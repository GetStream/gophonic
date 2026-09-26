// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3asr

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/GetStream/gophonic/speech"
)

// Exercise the public actor handoff with different prefix lengths, lanes that
// finish immediately, shrinking batches, draft verification, and repeated use.
// Separate scalar lanes follow the same call history as the batched lanes.
func TestGPUBatchTranscribeMatchesIndependent(t *testing.T) {
	m := loadModel(t, "gpu-q8")
	batch, err := NewBatchTranscriber(m, 8, LaneOptions{Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer batch.Close()
	var scalar [8]*Transcriber
	var got, want, previous [8]speech.Transcript
	for i := range scalar {
		scalar[i], err = NewTranscriber(m, LaneOptions{Threads: 1})
		if err != nil {
			t.Fatal(err)
		}
		defer scalar[i].Close()
	}
	jfk, zh := clipPCM(t, "jfk"), clipPCM(t, "zh")
	for round, count := range []int{2, 4, 8, 3, 1, 4, 4, 3} {
		var inputs [8][]float32
		var opts [8]speech.Options
		for i := range count {
			inputs[i] = jfk
			if i%2 != 0 {
				inputs[i] = zh
			}
			if round == 2 && i%3 == 0 {
				inputs[i] = nil
			}
			opts[i].Segments = i%2 == 0
			opts[i].Turn = true
			if round == 3 {
				opts[i].Context = "Names mentioned in this recording: Kennedy."
				if i%2 == 0 {
					opts[i].Language = speech.English
				} else {
					opts[i].Language = speech.Chinese
				}
			}
			if round == 5 {
				opts[i].Partial = &previous[i]
			}
			if round == 6 {
				opts[i].Languages = speech.Languages(speech.English, speech.Chinese)
			}
			if round == 7 && i == 0 {
				opts[i].Language = speech.English // mixed constrained/unrestricted lanes
			}
			if err := scalar[i].Transcribe(context.Background(), inputs[i], opts[i], &want[i]); err != nil {
				t.Fatal(err)
			}
		}
		// Native numeric arenas must remain owned across collections.
		runtime.GC()
		if err := batch.Transcribe(context.Background(), inputs[:count], opts[:count], got[:count]); err != nil {
			t.Fatal(err)
		}
		for i := range count {
			if !bytes.Equal(got[i].Text, want[i].Text) || got[i].Language != want[i].Language || got[i].Turn != want[i].Turn || !slices.Equal(got[i].Segments, want[i].Segments) || !slices.Equal(batch.lanes[i].tr.gen, scalar[i].gen) {
				t.Fatalf("round %d lane %d: batch %q (%v), scalar %q (%v); token or segment mismatch", round, i, got[i].Text, got[i].Language, want[i].Text, want[i].Language)
			}
			// The greedy batch path returns only the selected token instead of
			// copying full vocabulary rows. Recompute both final logit rows
			// outside Transcribe to retain a bit-exact state comparison.
			for _, tr := range []*Transcriber{batch.lanes[i].tr, scalar[i]} {
				if err := m.eval.LogitsInto(tr.hidden, tr.logits, tr.lm); err != nil {
					t.Fatal(err)
				}
			}
			if continuationFingerprint(batch.lanes[i].tr, &got[i]) != continuationFingerprint(scalar[i], &want[i]) {
				t.Fatalf("round %d lane %d: batch changed a float bit in encoder, hidden, logits, or draft verification", round, i)
			}
			previous[i].Text = append(previous[i].Text[:0], got[i].Text...)
			previous[i].Language = got[i].Language
		}
	}
}

// Cancel from the context check at a decode boundary. The counter is atomic
// because independent workers may inspect the same context concurrently.
type cancelDuringBatch struct {
	context.Context
	cancel context.CancelFunc
	checks atomic.Int32
}

func (c *cancelDuringBatch) Err() error {
	if c.checks.Add(1) >= 40 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestGPUBatchTranscribeErrorsAndReuse(t *testing.T) {
	m := loadModel(t, "gpu-q8")
	batch, err := NewBatchTranscriber(m, 4, LaneOptions{Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer batch.Close()
	pcm := clipPCM(t, "jfk")
	inputs := [][]float32{pcm, pcm, pcm, pcm}
	out := make([]speech.Transcript, len(inputs))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := batch.Transcribe(ctx, inputs, nil, out); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancel: %v", err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	mid := &cancelDuringBatch{Context: ctx, cancel: cancel}
	defer cancel()
	if err := batch.Transcribe(mid, inputs, nil, out); !errors.Is(err, context.Canceled) {
		t.Fatalf("decode cancellation: %v", err)
	}
	if mid.checks.Load() < 40 {
		t.Fatal("decode cancellation boundary was not reached")
	}
	options := make([]speech.Options, len(inputs))
	options[1].Words = true
	if err := batch.Transcribe(context.Background(), inputs, options, out); !errors.Is(err, speech.ErrUnsupported) {
		t.Fatalf("lane error: %v", err)
	}
	if err := batch.Transcribe(context.Background(), inputs, nil, out); err != nil {
		t.Fatalf("recovery: %v", err)
	}
	for i := 1; i < len(out); i++ {
		if !bytes.Equal(out[i].Text, out[0].Text) || out[i].Language != out[0].Language {
			t.Fatalf("recovery lane %d changed transcript", i)
		}
	}
	if err := batch.Transcribe(context.Background(), nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := batch.Transcribe(nil, inputs, nil, out); err == nil {
		t.Fatal("nil context accepted")
	}
	if err := batch.Transcribe(context.Background(), inputs, nil, out[:3]); err == nil {
		t.Fatal("short output accepted")
	}
	if err := batch.Transcribe(context.Background(), inputs, options[:3], out); err == nil {
		t.Fatal("short options accepted")
	}
	options[0].Partial = &out[3]
	if err := batch.Transcribe(context.Background(), inputs, options, out); err == nil {
		t.Fatal("aliased partial accepted")
	}
	aliased := make([]speech.Transcript, len(inputs))
	textStorage := make([]byte, 32)
	aliased[0].Text, aliased[1].Text = textStorage[:0], textStorage[4:4]
	if err := batch.Transcribe(context.Background(), inputs, nil, aliased); err == nil {
		t.Fatal("overlapping output capacities accepted")
	}
	aliased[1].Text = nil
	partialCopy := speech.Transcript{Text: textStorage[4:8]}
	clear(options)
	options[2].Partial = &partialCopy
	if err := batch.Transcribe(context.Background(), inputs, options, aliased); err == nil {
		t.Fatal("shallow partial/output alias accepted")
	}
	lanes := batch.lanes
	if err := batch.Close(); err != nil {
		t.Fatal(err)
	}
	if batch.m != nil || batch.lanes != nil || batch.decode != nil || batch.messages != nil {
		t.Fatal("closed batch retains model, workers, or decoder scratch")
	}
	for _, lane := range lanes {
		if lane.tr.m != nil || lane.tr.frontend != nil || lane.tr.lm != nil || lane.tr.genc != nil || lane.tr.vlogits != nil || lane.tr.decodeMemory != nil {
			t.Fatal("closed lane retains model or high-water scratch")
		}
	}
	if err := batch.Close(); err != nil {
		t.Fatal(err)
	}
	if err := batch.Transcribe(context.Background(), inputs, nil, out); !errors.Is(err, speech.ErrClosed) {
		t.Fatalf("closed: %v", err)
	}
}

// Same fixtures, warmup, and wave definition as the independent-worker
// benchmark. Transcript comparisons remain inside both timed loops.
func BenchmarkGPUBatchTranscribe(b *testing.B) {
	m := loadModel(b, "gpu-q8")
	pcm := clipPCM(b, "jfk")
	for _, count := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("lanes=%d", count), func(b *testing.B) {
			batch, err := NewBatchTranscriber(m, count, LaneOptions{Threads: 1})
			if err != nil {
				b.Fatal(err)
			}
			defer batch.Close()
			inputs := make([][]float32, count)
			out := make([]speech.Transcript, count)
			want := make([][]byte, count)
			for i := range count {
				inputs[i] = pcm
				if err := batch.lanes[i].tr.Transcribe(context.Background(), pcm, speech.Options{}, &out[i]); err != nil {
					b.Fatal(err)
				}
				want[i] = slices.Clone(out[i].Text)
			}
			run := func() {
				if err := batch.Transcribe(context.Background(), inputs, nil, out); err != nil {
					b.Fatal(err)
				}
				for i := range count {
					if !bytes.Equal(out[i].Text, want[i]) {
						b.Fatalf("batch lane %d transcript changed", i)
					}
				}
			}
			for range 2 {
				run()
			}
			b.ReportAllocs()
			for b.Loop() {
				run()
			}
			b.ReportMetric(float64(b.N*count)/b.Elapsed().Seconds(), "calls/s")
			b.ReportMetric(float64(count), "calls/op")
		})
	}
}

// BenchmarkGPUPairedTranscribe alternates baseline/batch order every pair to
// control clock, temperature, and desktop-load drift. A wave completes one
// full PCM-to-text call per lane; ns/op is the whole pair, not one inference.
// Reported allocation counts cover both methods together.
func BenchmarkGPUPairedTranscribe(b *testing.B) {
	m := loadModel(b, "gpu-q8")
	jfk, zh := clipPCM(b, "jfk"), clipPCM(b, "zh")
	for _, workload := range []string{"jfk", "mixed"} {
		for _, count := range []int{1, 2, 4, 8} {
			b.Run(fmt.Sprintf("%s/lanes=%d", workload, count), func(b *testing.B) {
				inputs := make([][]float32, count)
				for i := range inputs {
					inputs[i] = jfk
					if workload == "mixed" && i%2 != 0 {
						inputs[i] = zh
					}
				}
				independent := newGPUConcurrentCalls(b, m, inputs)
				batch, err := NewBatchTranscriber(m, count, LaneOptions{Threads: 1})
				if err != nil {
					b.Fatal(err)
				}
				defer batch.Close()
				out := make([]speech.Transcript, count)
				for i := range inputs {
					if err := batch.lanes[i].tr.Transcribe(context.Background(), inputs[i], speech.Options{}, &out[i]); err != nil {
						b.Fatal(err)
					}
				}
				runBatch := func() {
					if err := batch.Transcribe(context.Background(), inputs, nil, out); err != nil {
						b.Fatal(err)
					}
					for i := range out {
						if !bytes.Equal(out[i].Text, independent.want[i]) {
							b.Fatalf("batch lane %d changed transcript", i)
						}
					}
				}
				for range 2 {
					runBatch()
				}
				cpuTime := func() time.Duration {
					var usage syscall.Rusage
					if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
						b.Fatal(err)
					}
					seconds := int64(usage.Utime.Sec) + int64(usage.Stime.Sec)
					micros := int64(usage.Utime.Usec) + int64(usage.Stime.Usec)
					return time.Duration(seconds)*time.Second + time.Duration(micros)*time.Microsecond
				}
				var independentTime, batchTime, independentCPU, batchCPU time.Duration
				measureIndependent := func() {
					cpuStart := cpuTime()
					start := time.Now()
					if err := independent.run(); err != nil {
						b.Fatal(err)
					}
					independentTime += time.Since(start)
					independentCPU += cpuTime() - cpuStart
				}
				measureBatch := func() {
					cpuStart := cpuTime()
					start := time.Now()
					runBatch()
					batchTime += time.Since(start)
					batchCPU += cpuTime() - cpuStart
				}
				b.ReportAllocs()
				pair := 0
				for b.Loop() {
					if pair%2 == 0 {
						measureIndependent()
						measureBatch()
					} else {
						measureBatch()
						measureIndependent()
					}
					pair++
				}
				b.ReportMetric(float64(independentTime)/float64(b.N)/1e6, "independent-ms/wave")
				b.ReportMetric(float64(batchTime)/float64(b.N)/1e6, "batch-ms/wave")
				b.ReportMetric(float64(independentTime)/float64(batchTime), "speedup")
				b.ReportMetric(float64(count*b.N)/batchTime.Seconds(), "batch-calls/s")
				b.ReportMetric(float64(independentCPU)/float64(count*b.N)/1e6, "independent-cpu-ms/call")
				b.ReportMetric(float64(batchCPU)/float64(count*b.N)/1e6, "batch-cpu-ms/call")
			})
		}
	}
}
