// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

func BenchmarkCPUContextPrompt(b *testing.B) {
	m, encoder, oracle := decoderBenchmarkFixture(b)
	for _, count := range []int{2, 64, 128} {
		b.Run(fmt.Sprint(count), func(b *testing.B) {
			s := NewDecoderScratch()
			prompt, output := make([]int, count), make([]int, count+1)
			copy(prompt, oracle.Prefix)
			for i := len(oracle.Prefix); i < count; i++ {
				prompt[i] = oracle.Tokens[(i-len(oracle.Prefix))%(len(oracle.Tokens)-1)]
			}
			if _, err := m.GreedyDecodeInto(encoder, prompt, output, s, 50256); err != nil {
				b.Fatal(err)
			}
			want := output[count]
			b.ReportAllocs()
			for b.Loop() {
				if _, err := m.GreedyDecodeInto(encoder, prompt, output, s, 50256); err != nil {
					b.Fatal(err)
				}
				if output[count] != want {
					b.Fatal("unstable next token")
				}
			}
		})
	}
}

func BenchmarkCPURepeatedAudio(b *testing.B) {
	m, err := Load(testmodels.Path(b, testmodels.WhisperTinyEN))
	if err != nil {
		b.Fatal(err)
	}
	tr, err := NewTranscriber(m)
	if err != nil {
		b.Fatal(err)
	}
	defer tr.Close()
	data, err := os.ReadFile("../testdata/whisper_jfk.pcm.f32le")
	if err != nil {
		b.Fatal(err)
	}
	pcm, err := readFloatFixture("../testdata/whisper_jfk.pcm.f32le", len(data)/4)
	if err != nil {
		b.Fatal(err)
	}
	long := make([]float32, 0, len(pcm)*3)
	for range 3 {
		long = append(long, pcm...)
	}
	text := make([]byte, 0, 8192)
	warm, err := tr.TranscribeInto(long, text)
	if err != nil {
		b.Fatal(err)
	}
	want := string(warm)
	b.ReportAllocs()
	for b.Loop() {
		got, err := tr.TranscribeInto(long, text)
		if err != nil {
			b.Fatal(err)
		}
		if string(got) != want {
			b.Fatal("unstable multiwindow transcript")
		}
	}
}

// TestCPUMemoryFootprint is opt-in so ordinary model tests do not keep extra
// lanes alive. Copy this unchanged to the baseline revision for comparisons.
func TestCPUMemoryFootprint(t *testing.T) {
	if os.Getenv("GOPHONIC_MEMORY_REPORT") != "1" {
		t.Skip("set GOPHONIC_MEMORY_REPORT=1")
	}
	heap := func() uint64 { runtime.GC(); var s runtime.MemStats; runtime.ReadMemStats(&s); return s.HeapAlloc }
	start := heap()
	m, err := Load(testmodels.Path(t, testmodels.WhisperTinyEN))
	if err != nil {
		t.Fatal(err)
	}
	loaded := heap()
	data, err := os.ReadFile("../testdata/whisper_jfk.pcm.f32le")
	if err != nil {
		t.Fatal(err)
	}
	pcm, err := readFloatFixture("../testdata/whisper_jfk.pcm.f32le", len(data)/4)
	if err != nil {
		t.Fatal(err)
	}
	var lanes [2]*Transcriber
	for i := range lanes {
		lanes[i], err = NewTranscriber(m)
		if err != nil {
			t.Fatal(err)
		}
		defer lanes[i].Close()
		if _, err = lanes[i].TranscribeWindowInto(pcm, make([]byte, 0, 2048)); err != nil {
			t.Fatal(err)
		}
		t.Logf("lanes=%d live_heap_mib=%.3f", i+1, float64(heap()-start)/(1<<20))
	}
	t.Logf("loaded_model_heap_mib=%.3f", float64(loaded-start)/(1<<20))
	runtime.KeepAlive(m)
	runtime.KeepAlive(lanes)
}
