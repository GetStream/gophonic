// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package mel implements the log-mel frontends of gophonic's speech models.
package mel

import (
	"errors"
	"math"

	"github.com/GetStream/gophonic/internal/resample"
)

const (
	// FFTSize is the STFT window of every Whisper-style frontend.
	FFTSize    = 400
	fftBins    = FFTSize/2 + 1
	hopLength  = 160
	featurePad = FFTSize / 2

	// TurnBins, TurnFrames, and TurnSamples describe the turn-detector
	// window: the latest eight seconds of 16 kHz audio as [80,800] log-mel.
	TurnBins    = 80
	TurnFrames  = 800
	TurnSamples = 8 * 16000
)

var (
	ErrInvalidAudio  = errors.New("mel: audio must contain complete mono or stereo PCM frames")
	ErrInvalidRate   = errors.New("mel: sample rate must be between 8 kHz and 96 kHz")
	ErrNonFinite     = errors.New("mel: audio contains a NaN or infinite sample")
	ErrFeatureBuffer = errors.New("mel: feature output buffer has the wrong size")
)

// Turn is the Smart Turn and TinyMel frontend: the latest eight seconds of
// mono PCM resampled to 16 kHz, normalized to zero mean and unit variance,
// then Whisper's 80-band log-mel in float64. Load fills the window;
// FeaturesInto runs every stage, or a caller with its own workers runs
// Normalize, PowerFrames, MelRows, LogMelRows, and Finish itself. A Turn
// belongs to one call lane.
type Turn struct {
	samples   []float32
	padded    []float64
	power     []float64
	mel       []float64
	logMel    []float64
	fftReal   [FFTSize]float64
	fftImag   [FFTSize]float64
	resampler resample.Filter
	bank      *bank[float64]
}

// NewTurn allocates the window and stage buffers.
func NewTurn() *Turn {
	return &Turn{
		samples: make([]float32, TurnSamples),
		padded:  make([]float64, TurnSamples+2*featurePad),
		power:   make([]float64, fftBins*TurnFrames),
		mel:     make([]float64, TurnBins*TurnFrames),
		logMel:  make([]float64, TurnBins*TurnFrames),
		bank:    bank64(TurnBins),
	}
}

// Samples returns the 16 kHz window written by Load.
func (w *Turn) Samples() []float32 { return w.samples }

// Load resamples interleaved mono or stereo PCM at 8–96 kHz into the 16 kHz
// window, right-aligned: short audio is left-padded with silence and long
// audio keeps its last eight seconds. Once a sample rate is warm, Load does
// not allocate.
func (w *Turn) Load(pcm []float32, sampleRate, channels int) error {
	if sampleRate < 8000 || sampleRate > 96000 {
		return ErrInvalidRate
	}
	if channels < 1 || channels > 2 || len(pcm) == 0 || len(pcm)%channels != 0 {
		return ErrInvalidAudio
	}
	frames := len(pcm) / channels
	fullOutput64 := (int64(frames)*16000 + int64(sampleRate)/2) / int64(sampleRate)
	if fullOutput64 < 1 {
		return ErrInvalidAudio
	}
	outCount := int(fullOutput64)
	if outCount > TurnSamples {
		outCount = TurnSamples
	}
	sourceFirst := 0
	if fullOutput64 > TurnSamples {
		if sampleRate == 16000 {
			sourceFirst = frames - outCount
		} else {
			outputStart := fullOutput64 - int64(outCount)
			firstBase := int(outputStart * int64(sampleRate) / 16000)
			sourceFirst = firstBase - resample.Radius
			if sourceFirst < 0 {
				sourceFirst = 0
			}
		}
	}
	for frame := sourceFirst; frame < frames; frame++ {
		for channel := 0; channel < channels; channel++ {
			x := pcm[frame*channels+channel]
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return ErrNonFinite
			}
		}
	}
	leftPad := TurnSamples - outCount
	clear(w.samples)
	if sampleRate == 16000 {
		first := frames - outCount
		for i := 0; i < outCount; i++ {
			w.samples[leftPad+i] = monoAt(pcm, first+i, channels)
		}
		return nil
	}
	w.resampler.Prepare(sampleRate)
	outStart := fullOutput64 - int64(outCount)
	rem := int64(sampleRate)
	denom := int64(16000)
	coefficients := w.resampler.Coefficients
	for i := 0; i < outCount; i++ {
		globalOut := outStart + int64(i)
		numerator := globalOut * int64(sampleRate)
		base := numerator / denom
		phase := int(globalOut % int64(w.resampler.Phases))
		coeff := coefficients[phase*resample.Taps : (phase+1)*resample.Taps]
		var value float64
		for tap := 0; tap < resample.Taps; tap++ {
			sourceFrame := base + int64(tap-resample.Radius+1)
			if sourceFrame >= 0 && sourceFrame < int64(frames) {
				value += float64(monoAt(pcm, int(sourceFrame), channels)) * coeff[tap]
			}
		}
		w.samples[leftPad+i] = float32(value)
		rem += int64(sampleRate)
		if rem >= denom {
			rem %= denom
		}
	}
	return nil
}

func monoAt(pcm []float32, frame, channels int) float32 {
	if channels == 1 {
		return pcm[frame]
	}
	return (pcm[frame*2] + pcm[frame*2+1]) * 0.5
}

// FeaturesInto writes the normalized row-major [80,800] log-mel features of
// the window loaded by Load. It runs every stage on the calling goroutine.
func (w *Turn) FeaturesInto(output []float32) error {
	if len(output) != TurnBins*TurnFrames {
		return ErrFeatureBuffer
	}
	w.Normalize()
	// Whisper uses a centered, reflect-padded 400-point power STFT. Its
	// 801st frame is dropped, leaving 800 frames for this model.
	w.PowerFrames(0, TurnFrames, nil, nil)
	w.MelRows(0, TurnBins)
	w.Finish(output, w.LogMelRows(0, TurnBins))
	return nil
}

// Normalize normalizes the window and fills reflect padding.
// The FFT and mel stages may read the resulting padded samples concurrently.
func (w *Turn) Normalize() {
	audio := w.samples
	var mean float32
	for _, x := range audio {
		mean += x
	}
	mean /= float32(TurnSamples)
	var variance float32
	for _, x := range audio {
		d := x - mean
		variance += d * d
	}
	variance /= float32(TurnSamples)
	denom := float32(math.Sqrt(float64(variance + 1e-7)))
	for i, x := range audio {
		value := (x - mean) / denom
		audio[i] = value
		w.padded[featurePad+i] = float64(value)
	}

	for i := 0; i < featurePad; i++ {
		w.padded[i] = float64(audio[featurePad-i])
		w.padded[featurePad+TurnSamples+i] = float64(audio[TurnSamples-2-i])
	}
}

// LogMelRows writes independent mel rows and returns their maximum.
// Reducing those maxima after a worker barrier preserves the serial result.
func (w *Turn) LogMelRows(from, to int) float64 {
	maxLog := math.Inf(-1)
	for i := from * TurnFrames; i < to*TurnFrames; i++ {
		value := w.mel[i]
		if value < 1e-10 {
			value = 1e-10
		}
		v := math.Log10(value)
		w.logMel[i] = v
		if v > maxLog {
			maxLog = v
		}
	}
	return maxLog
}

// Finish applies Whisper's dynamic-range floor and output scaling once every
// log row is written; maxLog is the maximum over all rows.
func (w *Turn) Finish(output []float32, maxLog float64) {
	floor := maxLog - 8
	for i, value := range w.logMel {
		if value < floor {
			value = floor
		}
		output[i] = float32((value + 4) * 0.25)
	}
}

// PowerFrames writes distinct frame columns of the frequency-major
// power buffer. Callers may run disjoint ranges concurrently with separate
// real/imaginary scratch arrays after Normalize; nil scratch uses the
// Turn's own, for the calling goroutine.
func (w *Turn) PowerFrames(from, to int, realPart, imaginaryPart *[FFTSize]float64) {
	if realPart == nil {
		realPart, imaginaryPart = &w.fftReal, &w.fftImag
	}
	for frame := from; frame < to; frame++ {
		start := frame * hopLength
		powerFrame(tables64, w.padded[start:start+FFTSize], realPart, imaginaryPart, w.power[frame:], TurnFrames)
	}
}

// MelRows writes independent mel-band rows. Each row retains the same
// bin summation order as the serial Whisper reference.
func (w *Turn) MelRows(from, to int) {
	for mel := from; mel < to; mel++ {
		row := w.mel[mel*TurnFrames : (mel+1)*TurnFrames]
		clear(row)
		for bin, weight := range w.bank.rows[mel] {
			if weight == 0 {
				continue
			}
			power := w.power[bin*TurnFrames : (bin+1)*TurnFrames]
			for frame, p := range power {
				row[frame] += p * weight
			}
		}
	}
}
