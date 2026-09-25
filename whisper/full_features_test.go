// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/thesyncim/vibejson"
)

func TestFullFeatureFrames(t *testing.T) {
	cases := []struct {
		samples int
		want    int
	}{
		{samples: 0, want: 3000},
		{samples: 1, want: 3000},
		{samples: 159, want: 3000},
		{samples: 160, want: 3001},
		{samples: 176000, want: 4100},
		{samples: 736000, want: 7600},
	}
	for _, tc := range cases {
		got, err := FullFeatureFrames(tc.samples)
		if err != nil {
			t.Fatalf("FullFeatureFrames(%d): %v", tc.samples, err)
		}
		if got != tc.want {
			t.Fatalf("FullFeatureFrames(%d) = %d, want %d", tc.samples, got, tc.want)
		}
	}
	if _, err := FullFeatureFrames(-1); err != errNegativeFullFeatureSamples {
		t.Fatalf("negative samples error = %v, want %v", err, errNegativeFullFeatureSamples)
	}
	if _, err := FullFeatureFrames(int(^uint(0) >> 1)); err != errFullFeatureSizeOverflow {
		t.Fatalf("oversized input error = %v, want %v", err, errFullFeatureSizeOverflow)
	}
}

func TestFullFeaturesIntoMatchesPinnedOpenAIOracle(t *testing.T) {
	type oracleSample struct {
		Mel   int     `json:"mel"`
		Frame int     `json:"frame"`
		Value float32 `json:"value"`
	}
	type oracleInput struct {
		PCMSamples int            `json:"pcm_samples"`
		Frames     int            `json:"frames"`
		Samples    []oracleSample `json:"samples"`
	}
	var oracle struct {
		SourceCommit  string                 `json:"source_commit"`
		PaddingSample int                    `json:"padding_samples"`
		SampleRate    int                    `json:"sample_rate"`
		Values        map[string]oracleInput `json:"values"`
	}
	fixturePath := filepath.Join("..", "testdata", "whisper", "full_features.oracle.json")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := vibejson.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.SourceCommit != "86098128c0b4f24f0e2aa2994de830614b474227" || oracle.PaddingSample != featureSamples || oracle.SampleRate != featureSampleRate {
		t.Fatalf("oracle provenance/config mismatch: commit=%q padding=%d rate=%d", oracle.SourceCommit, oracle.PaddingSample, oracle.SampleRate)
	}

	jfk := readFeatureFloat32Fixture(t, "../testdata/whisper_jfk.pcm.f32le")
	inputs := map[string][]float32{
		"jfk":                  jfk,
		"jfk_plus_35s_silence": append(append([]float32(nil), jfk...), make([]float32, 35*featureSampleRate)...),
	}
	w := NewFullFeatureWorkspace()
	defer w.Close()
	for name, pcm := range inputs {
		t.Run(name, func(t *testing.T) {
			ref, ok := oracle.Values[name]
			if !ok {
				t.Fatalf("oracle fixture has no input %q", name)
			}
			if len(pcm) != ref.PCMSamples {
				t.Fatalf("input has %d PCM samples, oracle expects %d", len(pcm), ref.PCMSamples)
			}
			frames, err := FullFeatureFrames(len(pcm))
			if err != nil {
				t.Fatal(err)
			}
			if frames != ref.Frames {
				t.Fatalf("frame count = %d, oracle expects %d", frames, ref.Frames)
			}
			got := make([]float32, MelBins*frames)
			if err := FullFeaturesInto(pcm, got, w); err != nil {
				t.Fatal(err)
			}
			for _, sample := range ref.Samples {
				actual := got[sample.Mel*frames+sample.Frame]
				if delta := math.Abs(float64(actual - sample.Value)); delta > 5e-5 {
					t.Errorf("mel[%d,%d] = %.9g, oracle %.9g (abs error %.3g)", sample.Mel, sample.Frame, actual, sample.Value, delta)
				}
			}
		})
	}
}

func TestFullFeaturesWhisperSilenceAndFullInputValidation(t *testing.T) {
	w := NewFullFeatureWorkspace()
	defer w.Close()
	frames, err := FullFeatureFrames(0)
	if err != nil {
		t.Fatal(err)
	}
	dst := make([]float32, MelBins*frames)
	if err := FullFeaturesInto(nil, dst, w); err != nil {
		t.Fatal(err)
	}
	for i, value := range dst {
		if value != -1.5 {
			t.Fatalf("silence feature[%d] = %g, want -1.5", i, value)
		}
	}
	if err := FullFeaturesInto(nil, dst[:len(dst)-1], w); err != errFullFeatureOutputSize {
		t.Fatalf("short output error = %v, want %v", err, errFullFeatureOutputSize)
	}
	if err := FullFeaturesInto([]float32{0, float32(math.Inf(1))}, make([]float32, MelBins*3000), w); err != errFeatureNonFinite {
		t.Fatalf("non-finite full input error = %v, want %v", err, errFeatureNonFinite)
	}
	w.Close()
	if err := FullFeaturesInto(nil, dst, w); err != errNilFeatureWorkspace {
		t.Fatalf("closed workspace error = %v, want %v", err, errNilFeatureWorkspace)
	}
}

func TestFullFeaturesIntoWarmCallHasNoAllocations(t *testing.T) {
	w := NewFullFeatureWorkspace()
	defer w.Close()
	dst := make([]float32, MelBins*MelFrames)
	if err := FullFeaturesInto(nil, dst, w); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(3, func() {
		if err := FullFeaturesInto(nil, dst, w); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm full-file Whisper frontend allocated %.1f objects", allocs)
	}
}
