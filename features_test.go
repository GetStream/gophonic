// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic_test

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/internal/mel"
)

func TestWhisperFeatureWorkspaceParity(t *testing.T) {
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	w := gophonic.NewWhisperFeatureWorkspace()
	defer w.Close()
	got := make([]float32, mel.TurnBins*mel.TurnFrames)
	if err := gophonic.ExtractWhisperFeaturesInto(pcm, 16000, 1, got, w); err != nil {
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

	// A non-16 kHz stereo path must match the frontend used directly, bit for bit.
	stereo := make([]float32, 48000*2)
	for i := 0; i < len(stereo)/2; i++ {
		stereo[2*i] = float32(math.Sin(float64(i) * 0.017))
		stereo[2*i+1] = float32(math.Cos(float64(i) * 0.023))
	}
	old := mel.NewTurn()
	oldFeatures := make([]float32, mel.TurnBins*mel.TurnFrames)
	if err := old.Load(stereo, 48000, 2); err != nil {
		t.Fatal(err)
	}
	if err := old.FeaturesInto(oldFeatures); err != nil {
		t.Fatal(err)
	}
	if err := gophonic.ExtractWhisperFeaturesInto(stereo, 48000, 2, got, w); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(oldFeatures[i]) {
			t.Fatalf("48 kHz stereo feature[%d] differs: %g vs %g", i, got[i], oldFeatures[i])
		}
	}
}

func TestWhisperFeatureWorkspaceWarmZeroAllocAndClose(t *testing.T) {
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	w := gophonic.NewWhisperFeatureWorkspace()
	dst := make([]float32, mel.TurnBins*mel.TurnFrames)
	if err := gophonic.ExtractWhisperFeaturesInto(pcm, 16000, 1, dst, w); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(10, func() {
		if err := gophonic.ExtractWhisperFeaturesInto(pcm, 16000, 1, dst, w); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm Whisper extraction allocated %.1f objects", allocs)
	}
	w.Close()
	w.Close()
	if err := gophonic.ExtractWhisperFeaturesInto(pcm, 16000, 1, dst, w); err == nil {
		t.Fatalf("closed workspace error = %v, want an error", err)
	}
	if err := gophonic.ExtractWhisperFeaturesInto(pcm, 16000, 1, dst, nil); err == nil {
		t.Fatalf("nil workspace error = %v, want an error", err)
	}
}

func readFloatFixture(t testing.TB, path string) []float32 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data)%4 != 0 {
		t.Fatalf("fixture %s has invalid float32 byte length %d", path, len(data))
	}
	values := make([]float32, len(data)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return values
}
