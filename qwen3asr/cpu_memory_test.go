// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"strconv"
	"strings"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
)

// Opt-in footprint report with identical code on the baseline and candidate.
func TestCPUMemoryFootprint(t *testing.T) {
	if os.Getenv("GOPHONIC_MEMORY_REPORT") != "1" {
		t.Skip("set GOPHONIC_MEMORY_REPORT=1")
	}
	heap := func() uint64 {
		runtime.GC()
		// Opt-in diagnostic: separate live payload from retained free Go pages.
		if os.Getenv("GOPHONIC_MEMORY_SCAVENGE") == "1" {
			debug.FreeOSMemory()
		}
		var s runtime.MemStats
		runtime.ReadMemStats(&s)
		return s.HeapAlloc
	}
	start := heap()
	m, err := Load(testmodels.Path(t, testmodels.Qwen3ASR), Options{Format: qwen3lm.WeightsF16})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	loaded := heap()
	pcm := clipPCM(t, "jfk")
	var lanes [2]*Transcriber
	for i := range lanes {
		lanes[i], err = NewTranscriber(m, LaneOptions{})
		if err != nil {
			t.Fatal(err)
		}
		defer lanes[i].Close()
		var dst speech.Transcript
		if err := lanes[i].Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
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
	if os.Getenv("GOPHONIC_VMMAP_REPORT") == "1" && runtime.GOOS == "darwin" {
		report, err := exec.Command("vmmap", "-summary", strconv.Itoa(os.Getpid())).CombinedOutput()
		if err != nil {
			t.Fatal(err, string(report))
		}
		t.Logf("vmmap summary:\n%s", report)
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
