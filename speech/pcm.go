// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech

import (
	"errors"
	"math"

	"github.com/GetStream/gophonic/internal/resample"
)

const (
	pcmMinSampleRate = 8000
	pcmMaxSampleRate = 96000
)

var (
	ErrInvalidAudio      = errors.New("speech: PCM must contain complete mono or stereo frames")
	ErrInvalidSampleRate = errors.New("speech: PCM sample rate must be between 8 kHz and 96 kHz")
	ErrNonFinite         = errors.New("speech: PCM contains a NaN or infinite sample")
	ErrOutputCapacity    = errors.New("speech: PCM output buffer is too small")
	ErrInputTooLarge     = errors.New("speech: PCM is too large to resample")
)

// Resampler stores the polyphase sinc coefficients used by Resample16kInto.
// Give each concurrent caller its own workspace. A warmed call at a previously
// prepared sample rate does not allocate.
type Resampler struct {
	filter resample.Filter
	closed bool
}

// NewResampler creates reusable scratch for PCM resampling. Coefficients
// are prepared lazily for the first sample rate passed to Resample16kInto.
func NewResampler() *Resampler { return &Resampler{} }

// Close releases cached coefficients and prevents further use. It is safe to
// call more than once, provided no resampling call is in progress.
func (w *Resampler) Close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	w.filter = resample.Filter{}
}

// Samples16k returns the number of mono 16 kHz samples needed for an
// interleaved input containing inputSamples float32 values. It rounds the
// resampled frame count to the nearest sample, matching the gophonic audio
// frontend. Empty input needs zero output samples.
func Samples16k(inputSamples, sampleRate, channels int) (int, error) {
	if sampleRate < pcmMinSampleRate || sampleRate > pcmMaxSampleRate {
		return 0, ErrInvalidSampleRate
	}
	if inputSamples < 0 || channels < 1 || channels > 2 || inputSamples%channels != 0 {
		return 0, ErrInvalidAudio
	}
	frames := inputSamples / channels
	if frames == 0 {
		return 0, nil
	}
	const outputRate = int64(SampleRate)
	rate := int64(sampleRate)
	maxInt64 := int64(^uint64(0) >> 1)
	if int64(frames) > (maxInt64-rate/2)/outputRate {
		return 0, ErrInputTooLarge
	}
	count := (int64(frames)*outputRate + rate/2) / rate
	if count > int64(^uint(0)>>1) {
		return 0, ErrInputTooLarge
	}
	return int(count), nil
}

// Resample16kInto downmixes interleaved mono or stereo float32 PCM sampled at
// 8–96 kHz into mono 16 kHz PCM. It writes the actual duration to dst[:n] and
// returns n. No padding, truncation, or waveform normalization is applied.
// The caller can use Samples16k to size dst. For already mono 16 kHz PCM,
// callers can pass their input directly to inference and skip this copy.
//
// The resampler uses the same normalized 32-tap windowed-sinc polyphase filter
// as gophonic's shared audio frontend. Input samples are checked for NaN and
// infinity before dst is modified. dst must not overlap pcm except when the
// input is already mono 16 kHz, where the operation is a validated copy.
// Warm calls for a sample rate already cached in w allocate no memory.
func (w *Resampler) Resample16kInto(pcm []float32, sampleRate, channels int, dst []float32) (int, error) {
	if w == nil || w.closed {
		return 0, ErrClosed
	}
	outCount, err := Samples16k(len(pcm), sampleRate, channels)
	if err != nil {
		return 0, err
	}
	if len(dst) < outCount {
		return 0, ErrOutputCapacity
	}
	for _, sample := range pcm {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return 0, ErrNonFinite
		}
	}
	if outCount == 0 {
		return 0, nil
	}
	if sampleRate == SampleRate {
		if channels == 1 {
			copy(dst[:outCount], pcm)
			return outCount, nil
		}
		for frame := 0; frame < outCount; frame++ {
			dst[frame] = float32(pcmMonoAt(pcm, frame, channels))
		}
		return outCount, nil
	}

	w.filter.Prepare(sampleRate)
	coefficients := w.filter.Coefficients
	frames := len(pcm) / channels
	dst = dst[:outCount]
	base, remainder, phase := 0, 0, 0
	for output := range dst {
		coeff := coefficients[phase*resample.Taps : (phase+1)*resample.Taps]
		var value float64
		for tap, weight := range coeff {
			offset := tap - resample.Radius + 1
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
		base += remainder / SampleRate
		remainder %= SampleRate
		phase++
		if phase == w.filter.Phases {
			phase = 0
		}
	}
	return outCount, nil
}

func pcmMonoAt(pcm []float32, frame, channels int) float64 {
	if channels == 1 {
		return float64(pcm[frame])
	}
	return (float64(pcm[frame*2]) + float64(pcm[frame*2+1])) * 0.5
}
