// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"os"
	"runtime"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
)

// Opt-in footprint report with identical code on the baseline and candidate.
func TestCPUMemoryFootprint(t *testing.T) {
	if os.Getenv("GOPHONIC_MEMORY_REPORT") != "1" {
		t.Skip("set GOPHONIC_MEMORY_REPORT=1")
	}
	heap := func() uint64 { runtime.GC(); var s runtime.MemStats; runtime.ReadMemStats(&s); return s.HeapAlloc }
	start := heap()
	m, err := Load(testmodels.Path(t, testmodels.Qwen3ASR), Options{Format: FormatF16})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Release()
	loaded := heap()
	pcm := clipPCM(t, "jfk")
	var lanes [2]*Transcriber
	for i := range lanes {
		lanes[i], err = NewTranscriber(m, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer lanes[i].Close()
		var dst speech.Transcript
		if err := lanes[i].Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
			t.Fatal(err)
		}
		t.Logf("lanes=%d live_heap_mib=%.3f", i+1, float64(heap()-start)/(1<<20))
	}
	t.Logf("loaded_model_heap_mib=%.3f", float64(loaded-start)/(1<<20))
	runtime.KeepAlive(m)
	runtime.KeepAlive(lanes)
}
