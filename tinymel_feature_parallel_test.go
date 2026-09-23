// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor

import (
	"math"
	"runtime"
	"strconv"
	"testing"
)

func TestTinyParallelFeaturesMatchSerial(t *testing.T) {
	previous := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(previous)
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	serial := NewWorkspace()
	defer serial.Close()
	want := make([]float32, melCount*frameCount)
	if err := ExtractWhisperFeatures16k(pcm, want, serial); err != nil {
		t.Fatal(err)
	}
	for helpers := 1; helpers <= 7; helpers++ {
		ws := NewTinyMelWorkspaceWithWorkers(helpers)
		if err := prepareAudio(pcm, 16000, 1, &ws.featureHost); err != nil {
			t.Fatal(err)
		}
		got := ws.featureHost.features
		if err := ws.computeFeatures16kParallel(ws.featureHost.audio.samples, got); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("helpers=%d feature[%d]=%g, serial=%g", helpers, i, got[i], want[i])
			}
		}
		ws.Close()
	}
}

func TestTinyParallelFeaturesNoAllocations(t *testing.T) {
	previous := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(previous)
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	ws := NewTinyMelWorkspaceWithWorkers(7)
	defer ws.Close()
	allocs := testing.AllocsPerRun(5, func() {
		if err := prepareAudio(pcm, 16000, 1, &ws.featureHost); err != nil {
			t.Fatal(err)
		}
		if err := ws.computeFeatures16kParallel(ws.featureHost.audio.samples, ws.featureHost.features); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("parallel feature extraction allocated %.1f objects per call", allocs)
	}
}

func BenchmarkTinyParallelFeatureExtraction(b *testing.B) {
	pcm := readFloatFixture(b, "testdata/tone.pcm.f32le")
	for _, helpers := range [...]int{0, 1, 3, 7} {
		b.Run("helpers"+strconv.Itoa(helpers), func(b *testing.B) {
			ws := NewTinyMelWorkspaceWithWorkers(helpers)
			defer ws.Close()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := prepareAudio(pcm, 16000, 1, &ws.featureHost); err != nil {
					b.Fatal(err)
				}
				if err := ws.computeFeatures16kParallel(ws.featureHost.audio.samples, ws.featureHost.features); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
