// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"math"
)

const (
	PCMOutputSampleRate = 16000
	pcmMinSampleRate    = 8000
	pcmMaxSampleRate    = 96000
	pcmSincRadius       = 16
	pcmSincTaps         = 2 * pcmSincRadius
)

var (
	ErrPCMInvalidAudio      = errors.New("whisper: PCM must contain complete mono or stereo frames")
	ErrPCMInvalidSampleRate = errors.New("whisper: PCM sample rate must be between 8 kHz and 96 kHz")
	ErrPCMNonFinite         = errors.New("whisper: PCM contains a NaN or infinite sample")
	ErrPCMOutputCapacity    = errors.New("whisper: PCM output buffer is too small")
	ErrPCMWorkspace         = errors.New("whisper: nil or closed PCM workspace")
	ErrPCMInputTooLarge     = errors.New("whisper: PCM is too large to resample")
)

// PCMWorkspace stores the polyphase sinc coefficients used by Resample16kInto.
// Give each concurrent caller its own workspace. A warmed call at a previously
// prepared sample rate does not allocate.
type PCMWorkspace struct {
	coefficients []float64
	sampleRate   int
	phases       int
	closed       bool
}

// NewPCMWorkspace creates reusable scratch for PCM resampling. Coefficients
// are prepared lazily for the first sample rate passed to Resample16kInto.
func NewPCMWorkspace() *PCMWorkspace { return &PCMWorkspace{} }

// Close releases cached coefficients and prevents further use. It is safe to
// call more than once, provided no resampling call is in progress.
func (w *PCMWorkspace) Close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	w.coefficients = nil
}

// PCM16kSamples returns the number of mono 16 kHz samples needed for an
// interleaved input containing inputSamples float32 values. It rounds the
// resampled frame count to the nearest sample, matching the gophonic audio
// frontend. Empty input needs zero output samples.
func PCM16kSamples(inputSamples, sampleRate, channels int) (int, error) {
	if sampleRate < pcmMinSampleRate || sampleRate > pcmMaxSampleRate {
		return 0, ErrPCMInvalidSampleRate
	}
	if inputSamples < 0 || channels < 1 || channels > 2 || inputSamples%channels != 0 {
		return 0, ErrPCMInvalidAudio
	}
	frames := inputSamples / channels
	if frames == 0 {
		return 0, nil
	}
	const outputRate = int64(PCMOutputSampleRate)
	rate := int64(sampleRate)
	maxInt64 := int64(^uint64(0) >> 1)
	if int64(frames) > (maxInt64-rate/2)/outputRate {
		return 0, ErrPCMInputTooLarge
	}
	count := (int64(frames)*outputRate + rate/2) / rate
	if count > int64(^uint(0)>>1) {
		return 0, ErrPCMInputTooLarge
	}
	return int(count), nil
}

// Resample16kInto downmixes interleaved mono or stereo float32 PCM sampled at
// 8–96 kHz into mono 16 kHz PCM. It writes the actual duration to dst[:n] and
// returns n. No padding, truncation, or waveform normalization is applied.
// The caller can use PCM16kSamples to size dst. For already mono 16 kHz PCM,
// callers can pass their input directly to inference and skip this copy.
//
// The resampler uses the same normalized 32-tap windowed-sinc polyphase filter
// as gophonic's shared audio frontend. Input samples are checked for NaN and
// infinity before dst is modified. dst must not overlap pcm except when the
// input is already mono 16 kHz, where the operation is a validated copy.
// Warm calls for a sample rate already cached in w allocate no memory.
func (w *PCMWorkspace) Resample16kInto(pcm []float32, sampleRate, channels int, dst []float32) (int, error) {
	if w == nil || w.closed {
		return 0, ErrPCMWorkspace
	}
	outCount, err := PCM16kSamples(len(pcm), sampleRate, channels)
	if err != nil {
		return 0, err
	}
	if len(dst) < outCount {
		return 0, ErrPCMOutputCapacity
	}
	for _, sample := range pcm {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return 0, ErrPCMNonFinite
		}
	}
	if outCount == 0 {
		return 0, nil
	}
	if sampleRate == PCMOutputSampleRate {
		if channels == 1 {
			copy(dst[:outCount], pcm)
			return outCount, nil
		}
		for frame := 0; frame < outCount; frame++ {
			dst[frame] = float32(pcmMonoAt(pcm, frame, channels))
		}
		return outCount, nil
	}

	w.prepareResampler(sampleRate)
	coefficients := w.coefficients
	frames := len(pcm) / channels
	dst = dst[:outCount]
	base, remainder, phase := 0, 0, 0
	for output := range dst {
		coeff := coefficients[phase*pcmSincTaps : (phase+1)*pcmSincTaps]
		var value float64
		for tap, weight := range coeff {
			offset := tap - pcmSincRadius + 1
			if offset < 0 && base < -offset {
				continue
			}
			sourceFrame := base + offset
			if sourceFrame >= 0 && sourceFrame < frames {
				value += pcmMonoAt(pcm, sourceFrame, channels) * weight
			}
		}
		dst[output] = float32(value)

		remainder += sampleRate
		base += remainder / PCMOutputSampleRate
		remainder %= PCMOutputSampleRate
		phase++
		if phase == w.phases {
			phase = 0
		}
	}
	return outCount, nil
}

func (w *PCMWorkspace) prepareResampler(sampleRate int) {
	if w.sampleRate == sampleRate && len(w.coefficients) != 0 {
		return
	}
	phases := PCMOutputSampleRate / pcmGCD(sampleRate, PCMOutputSampleRate)
	needed := phases * pcmSincTaps
	if cap(w.coefficients) < needed {
		w.coefficients = make([]float64, needed)
	} else {
		w.coefficients = w.coefficients[:needed]
	}

	cutoff := float64(PCMOutputSampleRate) / float64(sampleRate)
	if cutoff > 1 {
		cutoff = 1
	}
	remainder := 0
	for phase := 0; phase < phases; phase++ {
		fraction := float64(remainder) / PCMOutputSampleRate
		var norm float64
		for tap := 0; tap < pcmSincTaps; tap++ {
			offset := tap - pcmSincRadius + 1
			distance := float64(offset) - fraction
			weight := cutoff * pcmSinc(cutoff*distance)
			if math.Abs(distance) < pcmSincRadius {
				weight *= 0.5 + 0.5*math.Cos(math.Pi*distance/pcmSincRadius)
			} else {
				weight = 0
			}
			w.coefficients[phase*pcmSincTaps+tap] = weight
			norm += weight
		}
		if norm != 0 {
			for tap := 0; tap < pcmSincTaps; tap++ {
				w.coefficients[phase*pcmSincTaps+tap] /= norm
			}
		}
		remainder = (remainder + sampleRate) % PCMOutputSampleRate
	}
	w.sampleRate = sampleRate
	w.phases = phases
}

func pcmMonoAt(pcm []float32, frame, channels int) float64 {
	if channels == 1 {
		return float64(pcm[frame])
	}
	return (float64(pcm[frame*2]) + float64(pcm[frame*2+1])) * 0.5
}

func pcmGCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func pcmSinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}
