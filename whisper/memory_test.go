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
	var w [2]*encoderWorkspace
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
	encoder, err := readFloatFixture("../testdata/whisper/jfk.encoder.f32le", audioFrames*audioState)
	if err != nil {
		t.Fatal(err)
	}
	s := newTinyDecoderScratch()
	if whispergemm.PackedVectorAccelerated() && len(s.crossKeys) != audioFrames*audioState {
		t.Fatal("raw cache retains more than one layer")
	}
	const first, second = 50257, 50362
	want := make([]float32, vocabSize)
	// Reuse the same scratch across full and short audio to catch stale packed
	// contents and layer scratch offsets after changing the cache shape.
	for _, frames := range []int{audioFrames, 37, audioFrames} {
		audio := encoder[:frames*audioState]
		if err := m.beginDecode(audio, s); err != nil {
			t.Fatal(err)
		}
		if err := m.logitsForTokenInto(first, 0, s, s.logits); err != nil {
			t.Fatal(err)
		}
		if err := m.logitsForTokenInto(second, 1, s, want); err != nil {
			t.Fatal(err)
		}
		if err := m.beginDecode(audio, s); err != nil {
			t.Fatal(err)
		}
		if err := m.decodeTokenInto(first, 0, s, nil, false); err != nil {
			t.Fatal(err)
		}
		runtime.GC()
		if err := m.logitsForTokenInto(second, 1, s, s.logits); err != nil {
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
