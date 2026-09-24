// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"math"
)

const (
	fullFeaturePowerFrames = 64
	fullFeatureRightPad    = featureSamples
)

var (
	errNegativeFullFeatureSamples = errors.New("whisper: full feature sample count must not be negative")
	errFullFeatureSizeOverflow    = errors.New("whisper: full feature input is too large")
	errFullFeatureOutputSize      = errors.New("whisper: full feature output has the wrong size")
)

// FullFeatureWorkspace owns reusable scratch for one concurrent full-file
// audio frontend. It must not be used by more than one call at a time.
type FullFeatureWorkspace struct {
	padded []float32
	power  []float32
	real   [featureFFTSize]float32
	imag   [featureFFTSize]float32
	closed bool
}

// NewFullFeatureWorkspace creates reusable scratch for FullFeaturesInto.
// The workspace grows to fit longer audio and retains that capacity for later
// calls. Allocate one workspace for each concurrent caller.
func NewFullFeatureWorkspace() *FullFeatureWorkspace {
	return &FullFeatureWorkspace{
		padded: make([]float32, featureSamples+2*featurePad),
		power:  make([]float32, featureFFTBins*fullFeaturePowerFrames),
	}
}

// Close releases the workspace buffers. Calls after Close return an error.
// Close must not race with FullFeaturesInto.
func (w *FullFeatureWorkspace) Close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	w.padded = nil
	w.power = nil
}

// FullFeatureFrames returns the number of Whisper log-mel frames produced for
// samples mono float32 PCM values sampled at 16 kHz. OpenAI Whisper appends
// thirty seconds of silence before its centered STFT and drops the final STFT
// frame, so the result is 3000 + floor(samples/160).
func FullFeatureFrames(samples int) (int, error) {
	if samples < 0 {
		return 0, errNegativeFullFeatureSamples
	}
	maxInt := int(^uint(0) >> 1)
	if samples > maxInt-(fullFeatureRightPad+2*featurePad) {
		return 0, errFullFeatureSizeOverflow
	}
	frames := MelFrames + samples/featureHopLength
	if frames > maxInt/MelBins {
		return 0, errFullFeatureSizeOverflow
	}
	return frames, nil
}

// FullFeaturesInto computes OpenAI Whisper's 80-band log-mel spectrogram for
// the entire mono float32 PCM input sampled at 16 kHz. It appends thirty
// seconds of silence, uses a centered 400-point STFT with a 160-sample hop,
// drops the final STFT frame, and applies max-minus-eight normalization over
// the complete spectrogram. dst is channel-major [80, frames], where frames is
// returned by FullFeatureFrames(len(pcm)).
//
// The frontend follows OpenAI Whisper source commit
// 86098128c0b4f24f0e2aa2994de830614b474227 and uses this package's shared FFT,
// Hann window, and 80-band Slaney mel bank. Once the workspace has grown for
// the input size, calls with the same or shorter audio allocate no memory.
// The workspace must be private to this call lane.
func FullFeaturesInto(pcm []float32, dst []float32, w *FullFeatureWorkspace) error {
	if w == nil || w.closed {
		return errNilFeatureWorkspace
	}
	frames, err := FullFeatureFrames(len(pcm))
	if err != nil {
		return err
	}
	if len(dst) != MelBins*frames {
		return errFullFeatureOutputSize
	}
	if len(w.padded) == 0 || cap(w.power) < featureFFTBins*fullFeaturePowerFrames {
		return errNilFeatureWorkspace
	}
	for _, x := range pcm {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return errFeatureNonFinite
		}
	}

	centerLength := len(pcm) + fullFeatureRightPad
	paddedLength := centerLength + 2*featurePad
	if cap(w.padded) < paddedLength {
		w.padded = make([]float32, paddedLength)
	} else {
		w.padded = w.padded[:paddedLength]
	}
	center := w.padded[featurePad : featurePad+centerLength]
	copy(center, pcm)
	clear(center[len(pcm):])
	for i := 0; i < featurePad; i++ {
		w.padded[i] = center[featurePad-i]
		w.padded[featurePad+centerLength+i] = center[centerLength-2-i]
	}

	maxLog := float32(math.Inf(-1))
	for firstFrame := 0; firstFrame < frames; {
		count := frames - firstFrame
		if count > fullFeaturePowerFrames {
			count = fullFeaturePowerFrames
		}
		if cap(w.power) < featureFFTBins*count {
			return errNilFeatureWorkspace
		}
		power := w.power[:featureFFTBins*count]
		for localFrame := 0; localFrame < count; localFrame++ {
			frame := firstFrame + localFrame
			start := frame * featureHopLength
			for n, index := range featureFFTOrder {
				w.real[index] = w.padded[start+n] * featureHann[n]
				w.imag[index] = 0
			}
			fftFeature400(&w.real, &w.imag)
			for bin := 0; bin < featureFFTBins; bin++ {
				re, im := w.real[bin], w.imag[bin]
				power[bin*count+localFrame] = re*re + im*im
			}
		}

		for mel := 0; mel < MelBins; mel++ {
			row := dst[mel*frames+firstFrame : mel*frames+firstFrame+count]
			clear(row)
			for bin, weight := range featureMelBank[mel] {
				if weight == 0 {
					continue
				}
				binPower := power[bin*count : (bin+1)*count]
				for localFrame, p := range binPower {
					row[localFrame] += p * weight
				}
			}
			for localFrame, value := range row {
				if value < 1e-10 {
					value = 1e-10
				}
				logMel := float32(math.Log10(float64(value)))
				row[localFrame] = logMel
				if logMel > maxLog {
					maxLog = logMel
				}
			}
		}
		firstFrame += count
	}

	floor := maxLog - 8
	for i, value := range dst {
		if value < floor {
			value = floor
		}
		dst[i] = (value + 4) / 4
	}
	return nil
}
