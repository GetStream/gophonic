// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package tinymel

import (
	"math"
	"runtime"
	"strconv"
	"testing"

	"github.com/GetStream/gophonic/internal/mel"
)

func TestTinyParallelFeaturesMatchSerial(t *testing.T) {
	previous := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(previous)
	pcm := readFloatFixture(t, "../testdata/tone.pcm.f32le")
	serial := mel.NewTurn()
	want := make([]float32, mel.TurnBins*mel.TurnFrames)
	if err := serial.Load(pcm, 16000, 1); err != nil {
		t.Fatal(err)
	}
	if err := serial.FeaturesInto(want); err != nil {
		t.Fatal(err)
	}
	for helpers := 1; helpers <= 7; helpers++ {
		ws := NewWorkspaceWithWorkers(helpers)
		if err := ws.audio.Load(pcm, 16000, 1); err != nil {
			t.Fatal(err)
		}
		got := ws.features
		if err := ws.computeFeaturesParallel(got); err != nil {
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
	pcm := readFloatFixture(t, "../testdata/tone.pcm.f32le")
	ws := NewWorkspaceWithWorkers(7)
	defer ws.Close()
	allocs := testing.AllocsPerRun(5, func() {
		if err := ws.audio.Load(pcm, 16000, 1); err != nil {
			t.Fatal(err)
		}
		if err := ws.computeFeaturesParallel(ws.features); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("parallel feature extraction allocated %.1f objects per call", allocs)
	}
}

func BenchmarkTinyParallelFeatureExtraction(b *testing.B) {
	pcm := readFloatFixture(b, "../testdata/tone.pcm.f32le")
	for _, helpers := range [...]int{0, 1, 3, 7} {
		b.Run("helpers"+strconv.Itoa(helpers), func(b *testing.B) {
			ws := NewWorkspaceWithWorkers(helpers)
			defer ws.Close()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := ws.audio.Load(pcm, 16000, 1); err != nil {
					b.Fatal(err)
				}
				if err := ws.computeFeaturesParallel(ws.features); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
