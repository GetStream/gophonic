// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func readFeatureFloat32Fixture(t *testing.T, name string) []float32 {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	if len(b)%4 != 0 {
		t.Fatalf("fixture %s has %d bytes, not a float32 array", name, len(b))
	}
	values := make([]float32, len(b)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return values
}

func TestFeaturesIntoMatchesPinnedOpenAIOracle(t *testing.T) {
	// Generated with the official OpenAI Whisper source and tiny.en checkpoint
	// pinned in testdata/whisper/jfk.oracle.json.
	pcm := readFeatureFloat32Fixture(t, "../testdata/whisper_jfk.pcm.f32le")
	want := readFeatureFloat32Fixture(t, "../testdata/whisper/jfk.mel.f32le")
	if len(want) != MelBins*MelFrames {
		t.Fatalf("oracle has %d values, want %d", len(want), MelBins*MelFrames)
	}
	w := NewFeatureWorkspace()
	defer w.Close()
	got := make([]float32, MelBins*MelFrames)
	if err := FeaturesInto(pcm, got, w); err != nil {
		t.Fatal(err)
	}
	var maxError, squaredError float64
	for i := range got {
		delta := math.Abs(float64(got[i] - want[i]))
		if delta > maxError {
			maxError = delta
		}
		squaredError += delta * delta
	}
	rmse := math.Sqrt(squaredError / float64(len(got)))
	t.Logf("pinned OpenAI frontend parity: max abs=%g, RMSE=%g", maxError, rmse)
	if maxError > 5e-5 || rmse > 2e-6 {
		t.Fatalf("features differ from pinned OpenAI oracle: max abs=%g, RMSE=%g (limits 5e-5, 2e-6)", maxError, rmse)
	}
}

func TestFeaturesWhisperSilenceAndPadding(t *testing.T) {
	w := NewFeatureWorkspace()
	defer w.Close()
	dst := make([]float32, MelBins*MelFrames)
	if err := FeaturesInto(nil, dst, w); err != nil {
		t.Fatal(err)
	}
	for i, v := range dst {
		if v != -1.5 {
			t.Fatalf("silence feature[%d] = %g, want -1.5", i, v)
		}
	}

	// A non-finite sample in the part truncated by pad_or_trim is not consumed.
	long := make([]float32, featureSamples+1)
	long[featureSamples] = float32(math.NaN())
	if err := FeaturesInto(long, dst, w); err != nil {
		t.Fatalf("truncated tail sample should be ignored: %v", err)
	}
	for i, v := range dst {
		if v != -1.5 {
			t.Fatalf("right-padded silence feature[%d] = %g, want -1.5", i, v)
		}
	}
}

func TestFeaturesRejectsInvalidWorkspaceBufferAndAudio(t *testing.T) {
	if err := FeaturesInto(nil, make([]float32, MelBins*MelFrames), nil); err != errNilFeatureWorkspace {
		t.Fatalf("nil workspace error = %v, want %v", err, errNilFeatureWorkspace)
	}
	w := NewFeatureWorkspace()
	dst := make([]float32, MelBins*MelFrames)
	if err := FeaturesInto(nil, dst[:len(dst)-1], w); err != errFeatureOutputSize {
		t.Fatalf("short output error = %v, want %v", err, errFeatureOutputSize)
	}
	if err := FeaturesInto([]float32{0, float32(math.Inf(1))}, dst, w); err != errFeatureNonFinite {
		t.Fatalf("non-finite audio error = %v, want %v", err, errFeatureNonFinite)
	}
	w.Close()
	w.Close()
	if err := FeaturesInto(nil, dst, w); err != errNilFeatureWorkspace {
		t.Fatalf("closed workspace error = %v, want %v", err, errNilFeatureWorkspace)
	}
}

func TestFeaturesIntoWarmCallHasNoAllocations(t *testing.T) {
	w := NewFeatureWorkspace()
	defer w.Close()
	pcm := make([]float32, 16000)
	for i := range pcm {
		pcm[i] = float32(0.2 * math.Sin(2*math.Pi*440*float64(i)/featureSampleRate))
	}
	dst := make([]float32, MelBins*MelFrames)
	if err := FeaturesInto(pcm, dst, w); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(3, func() {
		if err := FeaturesInto(pcm, dst, w); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm Whisper frontend allocated %.1f objects", allocs)
	}
}
