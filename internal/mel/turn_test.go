// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package mel

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func TestSilenceFeatures(t *testing.T) {
	pcm := make([]float32, TurnSamples)
	features := make([]float32, TurnBins*TurnFrames)
	ws := NewTurn()
	if err := extract(ws, pcm, features); err != nil {
		t.Fatal(err)
	}
	for i, value := range features {
		if value != -1.5 {
			t.Fatalf("feature %d = %g, want -1.5", i, value)
		}
	}
}

func TestToneFeaturesMatchWhisperOracle(t *testing.T) {
	pcm := readFloatFixture(t, "../../testdata/tone.pcm.f32le")
	want := readFloatFixture(t, "../../testdata/tone.mel.f32le")
	got := make([]float32, TurnBins*TurnFrames)
	ws := NewTurn()
	if err := extract(ws, pcm, got); err != nil {
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
	var samples, inputReal, inputImag [FFTSize]float64
	for n := 0; n < FFTSize; n++ {
		samples[n] = math.Sin(float64(n)*0.173) + 0.3*math.Cos(float64(n)*0.071)
		inputReal[fftOrder[n]] = samples[n]
	}
	fft400(tables64, &inputReal, &inputImag)
	for bin := 0; bin < FFTSize; bin++ {
		var wantReal, wantImag float64
		for n, sample := range samples {
			angle := -2 * math.Pi * float64(bin*n) / FFTSize
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
	pcm := readFloatFixture(t, "../../testdata/tone.pcm.f32le")
	features := make([]float32, TurnBins*TurnFrames)
	ws := NewTurn()
	if err := extract(ws, pcm, features); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if err := extract(ws, pcm, features); err != nil {
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
	ws := NewTurn()
	if err := ws.Load(pcm, 8000, 2); err != nil {
		t.Fatal(err)
	}
	centerSample := ws.Samples()[TurnSamples-8000]
	if math.Abs(float64(centerSample-0.3)) > 1e-6 {
		t.Fatalf("resampled stereo center = %.8f, want 0.3", centerSample)
	}
}

func TestAudioOutsideLastEightSecondsIsIgnored(t *testing.T) {
	pcm := make([]float32, TurnSamples+16000)
	pcm[0] = float32(math.NaN())
	ws := NewTurn()
	if err := ws.Load(pcm, 16000, 1); err != nil {
		t.Fatalf("audio outside the retained window should be ignored: %v", err)
	}
	pcm[TurnSamples] = float32(math.NaN())
	if err := ws.Load(pcm, 16000, 1); err != ErrNonFinite {
		t.Fatalf("NaN in the retained window error = %v, want %v", err, ErrNonFinite)
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

func extract(w *Turn, pcm, dst []float32) error {
	if err := w.Load(pcm, 16000, 1); err != nil {
		return err
	}
	return w.FeaturesInto(dst)
}

func TestFloat32PeriodicHannAndRadix400(t *testing.T) {
	var realPart, imaginaryPart [FFTSize]float32
	const bin = 8
	for n, index := range fftOrder {
		realPart[index] = float32(math.Cos(2 * math.Pi * bin * float64(n) / FFTSize))
	}
	fft400(tables32, &realPart, &imaginaryPart)
	for _, k := range []int{bin, FFTSize - bin} {
		if math.Abs(float64(realPart[k]-200)) > 0.01 || math.Abs(float64(imaginaryPart[k])) > 0.01 {
			t.Fatalf("FFT[%d] = (%g,%g), want (200,0)", k, realPart[k], imaginaryPart[k])
		}
	}
	if tables32.hann[0] != 0 || tables32.hann[FFTSize-1] == 0 {
		t.Fatalf("window is not periodic Hann: first=%g last=%g", tables32.hann[0], tables32.hann[FFTSize-1])
	}
}
