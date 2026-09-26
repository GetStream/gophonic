// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

func BenchmarkCPUContextPrompt(b *testing.B) {
	m, encoder, oracle := decoderBenchmarkFixture(b)
	for _, count := range []int{2, 64, 128} {
		b.Run(fmt.Sprint(count), func(b *testing.B) {
			s := newTinyDecoderScratch()
			prompt, output := make([]int, count), make([]int, count+1)
			copy(prompt, oracle.Prefix)
			for i := len(oracle.Prefix); i < count; i++ {
				prompt[i] = oracle.Tokens[(i-len(oracle.Prefix))%(len(oracle.Tokens)-1)]
			}
			if _, err := m.greedyDecodeInto(encoder, prompt, output, s, 50256); err != nil {
				b.Fatal(err)
			}
			want := output[count]
			b.ReportAllocs()
			for b.Loop() {
				if _, err := m.greedyDecodeInto(encoder, prompt, output, s, 50256); err != nil {
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
	tr, err := NewTranscriber(m, LaneOptions{})
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
		lanes[i], err = NewTranscriber(m, LaneOptions{})
		if err != nil {
			t.Fatal(err)
		}
		defer lanes[i].Close()
		if _, err = lanes[i].TranscribeWindowInto(pcm, make([]byte, 0, 2048)); err != nil {
			t.Fatal(err)
		}
		t.Logf("lanes=%d live_heap_mib=%.3f", i+1, float64(heap()-start)/(1<<20))
		// Peak RSS includes model-load transients; sample the live process too.
		if runtime.GOOS == "darwin" || runtime.GOOS == "linux" {
			rss, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("lanes=%d live_rss_kib=%s", i+1, strings.TrimSpace(string(rss)))
		}
	}
	t.Logf("loaded_model_heap_mib=%.3f", float64(loaded-start)/(1<<20))
	if path := os.Getenv("GOPHONIC_HEAP_PROFILE"); path != "" {
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := pprof.WriteHeapProfile(f); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
	runtime.KeepAlive(m)
	runtime.KeepAlive(lanes)
}
