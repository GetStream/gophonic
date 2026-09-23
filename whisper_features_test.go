// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor

import (
	"math"
	"testing"
)

func TestWhisperFeatureWorkspaceParity(t *testing.T) {
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	w := NewWhisperFeatureWorkspace()
	defer w.Close()
	got := make([]float32, melCount*frameCount)
	if err := ExtractWhisperFeaturesInto(pcm, 16000, 1, got, w); err != nil {
		t.Fatal(err)
	}
	want := readFloatFixture(t, "testdata/tone.mel.f32le")
	maxError := 0.0
	for i := range got {
		error := math.Abs(float64(got[i] - want[i]))
		if error > maxError {
			maxError = error
		}
	}
	if maxError > 2e-5 {
		t.Fatalf("Whisper oracle max error %g exceeds 2e-5", maxError)
	}

	// A non-16 kHz stereo path must match the existing frontend bit for bit.
	stereo := make([]float32, 48000*2)
	for i := 0; i < len(stereo)/2; i++ {
		stereo[2*i] = float32(math.Sin(float64(i) * 0.017))
		stereo[2*i+1] = float32(math.Cos(float64(i) * 0.023))
	}
	old := NewWorkspace()
	defer old.Close()
	if err := prepareAudio(stereo, 48000, 2, old); err != nil {
		t.Fatal(err)
	}
	if err := computeFeatures16k(old.audio.samples, old.features, &old.audio); err != nil {
		t.Fatal(err)
	}
	if err := ExtractWhisperFeaturesInto(stereo, 48000, 2, got, w); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(old.features[i]) {
			t.Fatalf("48 kHz stereo feature[%d] differs: %g vs %g", i, got[i], old.features[i])
		}
	}
}

func TestWhisperFeatureWorkspaceWarmZeroAllocAndClose(t *testing.T) {
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	w := NewWhisperFeatureWorkspace()
	dst := make([]float32, melCount*frameCount)
	if err := ExtractWhisperFeaturesInto(pcm, 16000, 1, dst, w); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(10, func() {
		if err := ExtractWhisperFeaturesInto(pcm, 16000, 1, dst, w); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm Whisper extraction allocated %.1f objects", allocs)
	}
	w.Close()
	w.Close()
	if err := ExtractWhisperFeaturesInto(pcm, 16000, 1, dst, w); err != errNilWorkspace {
		t.Fatalf("closed workspace error = %v, want %v", err, errNilWorkspace)
	}
	if err := ExtractWhisperFeaturesInto(pcm, 16000, 1, dst, nil); err != errNilWorkspace {
		t.Fatalf("nil workspace error = %v, want %v", err, errNilWorkspace)
	}
}
