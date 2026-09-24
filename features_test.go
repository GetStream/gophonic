// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func TestSilenceFeatures(t *testing.T) {
	pcm := make([]float32, maxWindowSamples)
	features := make([]float32, melCount*frameCount)
	ws := NewWorkspace()
	defer ws.Close()
	if err := ExtractWhisperFeatures16k(pcm, features, ws); err != nil {
		t.Fatal(err)
	}
	for i, value := range features {
		if value != -1.5 {
			t.Fatalf("feature %d = %g, want -1.5", i, value)
		}
	}
}

func TestToneFeaturesMatchWhisperOracle(t *testing.T) {
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	want := readFloatFixture(t, "testdata/tone.mel.f32le")
	got := make([]float32, melCount*frameCount)
	ws := NewWorkspace()
	defer ws.Close()
	if err := ExtractWhisperFeatures16k(pcm, got, ws); err != nil {
		t.Fatal(err)
	}
	var maxError, squaredError float64
	for i := range want {
		delta := math.Abs(float64(got[i] - want[i]))
		if delta > maxError {
			maxError = delta
		}
		squaredError += delta * delta
	}
	rmse := math.Sqrt(squaredError / float64(len(want)))
	if maxError > 2e-5 || rmse > 2e-6 {
		t.Fatalf("Whisper feature mismatch: max abs %.9g, RMSE %.9g", maxError, rmse)
	}
}

func TestFFT400MatchesReferenceDFT(t *testing.T) {
	var samples, inputReal, inputImag [fftSize]float64
	for n := 0; n < fftSize; n++ {
		samples[n] = math.Sin(float64(n)*0.173) + 0.3*math.Cos(float64(n)*0.071)
		inputReal[fftOrder[n]] = samples[n]
	}
	fft400(&inputReal, &inputImag)
	for bin := 0; bin < fftSize; bin++ {
		var wantReal, wantImag float64
		for n, sample := range samples {
			angle := -2 * math.Pi * float64(bin*n) / fftSize
			wantReal += sample * math.Cos(angle)
			wantImag += sample * math.Sin(angle)
		}
		errorReal := inputReal[bin] - wantReal
		errorImag := inputImag[bin] - wantImag
		if math.Hypot(errorReal, errorImag) > 1e-9*math.Max(1, math.Hypot(wantReal, wantImag)) {
			t.Fatalf("FFT bin %d mismatch: got %.12g%+.12gi, want %.12g%+.12gi", bin, inputReal[bin], inputImag[bin], wantReal, wantImag)
		}
	}
}

func TestFeatureExtractionHasNoSteadyStateAllocations(t *testing.T) {
	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	features := make([]float32, melCount*frameCount)
	ws := NewWorkspace()
	defer ws.Close()
	if err := ExtractWhisperFeatures16k(pcm, features, ws); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if err := ExtractWhisperFeatures16k(pcm, features, ws); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("steady-state feature extraction allocated %.1f objects per call, want zero", allocs)
	}
}

func TestStereoResamplingProducesExpectedCenterSample(t *testing.T) {
	pcm := make([]float32, 8000*2)
	for frame := 0; frame < 8000; frame++ {
		pcm[frame*2], pcm[frame*2+1] = 0.2, 0.4
	}
	ws := NewWorkspace()
	defer ws.Close()
	if err := prepareAudio(pcm, 8000, 2, ws); err != nil {
		t.Fatal(err)
	}
	centerSample := ws.audio.samples[maxWindowSamples-8000]
	if math.Abs(float64(centerSample-0.3)) > 1e-6 {
		t.Fatalf("resampled stereo center = %.8f, want 0.3", centerSample)
	}
}

func TestAudioOutsideLastEightSecondsIsIgnored(t *testing.T) {
	pcm := make([]float32, maxWindowSamples+16000)
	pcm[0] = float32(math.NaN())
	ws := NewWorkspace()
	defer ws.Close()
	if err := prepareAudio(pcm, 16000, 1, ws); err != nil {
		t.Fatalf("audio outside the retained window should be ignored: %v", err)
	}
	pcm[maxWindowSamples] = float32(math.NaN())
	if err := prepareAudio(pcm, 16000, 1, ws); err != errNonFiniteAudio {
		t.Fatalf("NaN in the retained window error = %v, want %v", err, errNonFiniteAudio)
	}
}

func TestPredictionMatchesOnnxOracle(t *testing.T) {
	modelPath := os.Getenv("GOPHONIC_TEST_MODEL")
	if modelPath == "" {
		t.Skip("set GOPHONIC_TEST_MODEL to a converted Smart Turn v3.2 .gophonic bundle to run ONNX parity checks")
	}
	model, err := Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	ws := NewWorkspace()
	defer ws.Close()
	zeroFeatures := make([]float32, melCount*frameCount)
	check := func(name string, features []float32, want float32) {
		t.Helper()
		got, err := model.PredictFeaturesInto(features, ws)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(got.Probability-want)) > 2e-5 {
			t.Errorf("%s probability = %.9f, want %.9f", name, got.Probability, want)
		}
	}
	check("zero features", zeroFeatures, 0.98987246)
	toneFeatures := readFloatFixture(t, "testdata/tone.mel.f32le")
	check("440 Hz tone", toneFeatures, 0.0663098394870758)
	patternFeatures := make([]float32, melCount*frameCount)
	for i := range patternFeatures {
		patternFeatures[i] = float32((i*73)%1009-504) / 256
	}
	check("deterministic feature pattern", patternFeatures, 0.97055656)
	allocs := testing.AllocsPerRun(3, func() {
		if _, err := model.PredictFeaturesInto(zeroFeatures, ws); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Errorf("steady-state PredictFeaturesInto allocated %.1f objects per call, want zero", allocs)
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
