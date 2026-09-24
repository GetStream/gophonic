// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"testing"
)

func TestPCM16kSamples(t *testing.T) {
	tests := []struct {
		name     string
		input    int
		rate     int
		channels int
		want     int
		wantErr  error
	}{
		{name: "empty", input: 0, rate: 48000, channels: 2, want: 0},
		{name: "same rate", input: 16000, rate: 16000, channels: 1, want: 16000},
		{name: "48k stereo", input: 96000, rate: 48000, channels: 2, want: 16000},
		{name: "round up", input: 5, rate: 8000, channels: 1, want: 10},
		{name: "short rounds to zero", input: 1, rate: 96000, channels: 1, want: 0},
		{name: "invalid rate", input: 1, rate: 7999, channels: 1, wantErr: ErrPCMInvalidSampleRate},
		{name: "invalid channels", input: 1, rate: 16000, channels: 3, wantErr: ErrPCMInvalidAudio},
		{name: "incomplete stereo frame", input: 3, rate: 48000, channels: 2, wantErr: ErrPCMInvalidAudio},
		{name: "negative count", input: -1, rate: 16000, channels: 1, wantErr: ErrPCMInvalidAudio},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := PCM16kSamples(tt.input, tt.rate, tt.channels)
			if err != tt.wantErr {
				t.Fatalf("error = %v, want %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("sample count = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestResample16kIntoCopiesMonoAndPreservesLongInput(t *testing.T) {
	const thirtySeconds = 30 * PCMOutputSampleRate
	pcm := make([]float32, thirtySeconds+17)
	pcm[0], pcm[thirtySeconds], pcm[len(pcm)-1] = -0.75, 0.125, 0.875
	dst := make([]float32, len(pcm)+1)
	dst[len(dst)-1] = 42
	w := NewPCMWorkspace()
	defer w.Close()

	n, err := w.Resample16kInto(pcm, 16000, 1, dst)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(pcm) {
		t.Fatalf("output samples = %d, want all %d input samples", n, len(pcm))
	}
	for _, index := range []int{0, thirtySeconds, len(pcm) - 1} {
		if dst[index] != pcm[index] {
			t.Fatalf("output[%d] = %g, want exact input %g", index, dst[index], pcm[index])
		}
	}
	if dst[n] != 42 {
		t.Fatalf("sample past output = %g, want untouched sentinel 42", dst[n])
	}
}

func TestResample16kIntoDownmixesStereo(t *testing.T) {
	pcm := []float32{0.2, 0.4, 0.2, 0.6, math.MaxFloat32, math.MaxFloat32}
	dst := make([]float32, 3)
	w := NewPCMWorkspace()
	defer w.Close()

	n, err := w.Resample16kInto(pcm, 16000, 2, dst)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("output samples = %d, want 3", n)
	}
	for i, want := range []float32{0.3, 0.4, math.MaxFloat32} {
		if math.Abs(float64(dst[i]-want)) > 1e-7*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("downmixed output[%d] = %g, want %g", i, dst[i], want)
		}
	}
}

func TestResample16kIntoHandles48kOpusPCM(t *testing.T) {
	// Opus decoders commonly provide float32 PCM at 48 kHz. A 100 ms 1 kHz
	// tone should become exactly 1600 samples and retain its level and phase.
	const frames = 4800
	pcm := make([]float32, frames)
	for i := range pcm {
		pcm[i] = float32(0.5 * math.Sin(2*math.Pi*1000*float64(i)/48000))
	}
	dst := make([]float32, 1600)
	w := NewPCMWorkspace()
	defer w.Close()

	n, err := w.Resample16kInto(pcm, 48000, 1, dst)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(dst) {
		t.Fatalf("output samples = %d, want %d", n, len(dst))
	}
	maxError := 0.0
	for i := 32; i < n-32; i++ {
		want := 0.5 * math.Sin(2*math.Pi*1000*float64(i)/16000)
		delta := math.Abs(float64(dst[i]) - want)
		if delta > maxError {
			maxError = delta
		}
	}
	if maxError > 5e-4 {
		t.Fatalf("48 kHz resample max tone error = %g, want <= 5e-4", maxError)
	}
}

func TestResample16kIntoRejectsInvalidInputWithoutWriting(t *testing.T) {
	w := NewPCMWorkspace()
	defer w.Close()
	for _, tt := range []struct {
		name     string
		pcm      []float32
		rate     int
		channels int
		dst      []float32
		wantErr  error
	}{
		{name: "too small output", pcm: []float32{0, 0}, rate: 16000, channels: 1, dst: []float32{9}, wantErr: ErrPCMOutputCapacity},
		{name: "nan", pcm: []float32{0, float32(math.NaN())}, rate: 16000, channels: 1, dst: []float32{9, 9}, wantErr: ErrPCMNonFinite},
		{name: "infinity", pcm: []float32{float32(math.Inf(-1))}, rate: 16000, channels: 1, dst: []float32{9}, wantErr: ErrPCMNonFinite},
		{name: "bad rate", pcm: []float32{0}, rate: 96001, channels: 1, dst: []float32{9}, wantErr: ErrPCMInvalidSampleRate},
		{name: "partial frame", pcm: []float32{0}, rate: 16000, channels: 2, dst: []float32{9}, wantErr: ErrPCMInvalidAudio},
	} {
		t.Run(tt.name, func(t *testing.T) {
			before := append([]float32(nil), tt.dst...)
			if _, err := w.Resample16kInto(tt.pcm, tt.rate, tt.channels, tt.dst); err != tt.wantErr {
				t.Fatalf("error = %v, want %v", err, tt.wantErr)
			}
			for i := range tt.dst {
				if tt.dst[i] != before[i] {
					t.Fatalf("error modified dst[%d] from %g to %g", i, before[i], tt.dst[i])
				}
			}
		})
	}
}

func TestPCMWorkspaceWarmResampleDoesNotAllocate(t *testing.T) {
	pcm := make([]float32, 48000)
	for i := range pcm {
		pcm[i] = float32(0.25 * math.Sin(2*math.Pi*440*float64(i)/48000))
	}
	dst := make([]float32, 16000)
	w := NewPCMWorkspace()
	defer w.Close()
	if _, err := w.Resample16kInto(pcm, 48000, 1, dst); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(3, func() {
		if _, err := w.Resample16kInto(pcm, 48000, 1, dst); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm PCM resampling allocated %.1f objects", allocs)
	}
}

func TestPCMWorkspaceClose(t *testing.T) {
	w := NewPCMWorkspace()
	w.Close()
	w.Close()
	if _, err := w.Resample16kInto(nil, 16000, 1, nil); err != ErrPCMWorkspace {
		t.Fatalf("closed workspace error = %v, want %v", err, ErrPCMWorkspace)
	}
	var nilWorkspace *PCMWorkspace
	if _, err := nilWorkspace.Resample16kInto(nil, 16000, 1, nil); err != ErrPCMWorkspace {
		t.Fatalf("nil workspace error = %v, want %v", err, ErrPCMWorkspace)
	}
}
