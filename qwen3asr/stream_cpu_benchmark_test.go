// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"bytes"
	"context"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

type cpuStreamTrace struct {
	tr        *Transcriber
	prev, cur speech.Transcript
}

func cpuGrowingAudio(t testing.TB, e *encoder) ([]float32, []int) {
	t.Helper()
	step := e.prefixStep()
	frames := []int{step / 4, step / 2, step + 2, step + 100, step + step/2, 2*step + 2, 2*step + 100}
	clip := clipPCM(t, "jfk")
	clip = clip[:len(clip)/160*160]
	pcm := make([]float32, frames[len(frames)-1]*160)
	for i := range pcm {
		pcm[i] = clip[i%len(clip)]
	}
	return pcm, frames
}

func (s *cpuStreamTrace) run(t testing.TB, pcm []float32, frames []int) {
	t.Helper()
	s.prev.Reset()
	s.cur.Reset()
	for _, n := range frames {
		if err := s.tr.Transcribe(context.Background(), pcm[:n*160], speech.Options{Partial: &s.prev}, &s.cur); err != nil {
			t.Fatal(err)
		}
		s.prev, s.cur = s.cur, s.prev
	}
}

// BenchmarkCPUStreaming measures a complete sequence of strictly growing
// audio inputs. Each new transcription verifies the previous partial, and
// completed encoder windows can be reused only after exact input comparison.
func BenchmarkCPUStreaming(b *testing.B) {
	m := loadModel(b, qwen3lm.WeightsF16)
	tr, err := NewTranscriber(m, LaneOptions{})
	if err != nil {
		b.Fatal(err)
	}
	defer tr.Close()
	pcm, frames := cpuGrowingAudio(b, m.enc)
	s := cpuStreamTrace{tr: tr}
	s.run(b, pcm, frames)
	want := slices.Clone(s.prev.Text)
	b.ReportAllocs()
	for b.Loop() {
		s.run(b, pcm, frames)
		if !bytes.Equal(s.prev.Text, want) {
			b.Fatal("streaming transcript changed")
		}
	}
	b.ReportMetric(float64(len(frames)), "calls/trace")
}
