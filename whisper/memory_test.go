// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"runtime"
	"sync"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

func TestEncoderAttentionArenaOwnershipAndClose(t *testing.T) {
	var lanes [2]*EncoderWorkspace
	for i := range lanes {
		var err error
		lanes[i], err = NewEncoderWorkspaceWithWorkers(8)
		if err != nil {
			t.Fatal(err)
		}
		defer lanes[i].Close()
	}
	first, second := lanes[0].attention, lanes[1].attention
	memory := first.memory
	// tiny.en: 8 score tiles and six pairs of packed K/V regions. All
	// payloads fit exactly; no per-region alignment padding is needed here.
	const wantBytes = 6150144
	if memory == nil || memory.Bytes() != wantBytes {
		t.Fatalf("attention arena bytes=%d, want %d", memory.Bytes(), wantBytes)
	}
	if memory == second.memory || &first.scores[0] == &second.scores[0] || first.keys[0] == second.keys[0] || first.values[0] == second.values[0] {
		t.Fatal("lanes share mutable attention storage")
	}
	first.scores[0] = 42
	runtime.GC()
	if first.scores[0] != 42 || second.scores[0] != 0 {
		t.Fatal("attention storage did not survive GC independently")
	}
	lanes[0].Close()
	if memory.Bytes() != 0 || first.memory != nil || first.scores != nil || first.keys != nil || first.values != nil {
		t.Fatal("Close retained attention payload or views")
	}
	lanes[0].Close()
	second.scores[0] = 7
	if second.memory.Bytes() != wantBytes || second.scores[0] != 7 {
		t.Fatal("closing one lane invalidated another")
	}
	runtime.KeepAlive(lanes)
	t.Logf("private attention arena: %d bytes; released on Close", wantBytes)
}

func TestLanesShareOnlyImmutableEncoderPacking(t *testing.T) {
	m, err := Load(testmodels.Path(t, testmodels.WhisperTinyEN))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	weights, ok := bindEncoderWeights(m)
	if !ok {
		t.Fatal("invalid weights")
	}
	var w [2]*EncoderWorkspace
	for i := range w {
		w[i], err = newEncoderWorkspace(m.dims, 1)
		if err != nil {
			t.Fatal(err)
		}
		defer w[i].Close()
	}
	var wg sync.WaitGroup
	for i := range w {
		wg.Go(func() {
			if err := w[i].preparePacked(m, weights); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if w[0].conv1Weight != w[1].conv1Weight || w[0].packed[0].mlpIn != w[1].packed[0].mlpIn {
		t.Fatal("duplicated immutable packing")
	}
	if &w[0].q[0] == &w[1].q[0] {
		t.Fatal("lanes share mutable activations")
	}
	runtime.GC()
	w[0].q[0] = 42
	if w[1].q[0] != 0 {
		t.Fatal("lane write escaped its arena")
	}
	w[0].Close()
	w[1].q[0] = 7
	runtime.KeepAlive(w)
}

func TestPromptWithoutLogitsPreservesDecoderState(t *testing.T) {
	m, err := Load(testmodels.Path(t, testmodels.WhisperTinyEN))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	encoder, err := readFloatFixture("../testdata/whisper/jfk.encoder.f32le", AudioFrames*AudioState)
	if err != nil {
		t.Fatal(err)
	}
	s := NewDecoderScratch()
	if whispergemm.PackedVectorAccelerated() && len(s.crossKeys) != AudioFrames*AudioState {
		t.Fatal("raw cache retains more than one layer")
	}
	const first, second = 50257, 50362
	want := make([]float32, VocabSize)
	// Reuse the same scratch across full and short audio to catch stale packed
	// contents and layer scratch offsets after changing the cache shape.
	for _, frames := range []int{AudioFrames, 37, AudioFrames} {
		audio := encoder[:frames*AudioState]
		if err := m.BeginDecode(audio, s); err != nil {
			t.Fatal(err)
		}
		if err := m.LogitsForTokenInto(first, 0, s, s.logits); err != nil {
			t.Fatal(err)
		}
		if err := m.LogitsForTokenInto(second, 1, s, want); err != nil {
			t.Fatal(err)
		}
		if err := m.BeginDecode(audio, s); err != nil {
			t.Fatal(err)
		}
		if err := m.decodeTokenInto(first, 0, s, nil, false); err != nil {
			t.Fatal(err)
		}
		runtime.GC()
		if err := m.LogitsForTokenInto(second, 1, s, s.logits); err != nil {
			t.Fatal(err)
		}
		for i, x := range want {
			if math.Float32bits(x) != math.Float32bits(s.logits[i]) {
				t.Fatalf("frames=%d logit %d changed", frames, i)
			}
		}
	}
}

func TestModelCloseReleasesArena(t *testing.T) {
	m, err := Load(testmodels.Path(t, testmodels.WhisperTinyEN))
	if err != nil {
		t.Fatal(err)
	}
	memory := m.memory
	if memory == nil || memory.Bytes() == 0 {
		t.Fatal("model lacks mapped storage")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if memory.Bytes() != 0 || m.tensors != nil {
		t.Fatal("Close retained mapped weights")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
}
