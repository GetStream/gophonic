// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package resample builds the polyphase filter that every gophonic frontend
// uses to bring PCM to 16 kHz.
package resample

import "math"

const (
	// Rate is the output sample rate.
	Rate = 16000
	// Radius is the filter half-width in input samples; Taps = 2*Radius.
	Radius = 16
	Taps   = 2 * Radius
)

// Filter holds normalized 32-tap Hann-windowed sinc coefficients for one
// input rate: Coefficients[phase*Taps+tap] weighs input frame
// base+tap-Radius+1 for output phase phase. The zero value is ready to use.
type Filter struct {
	Coefficients []float64
	Phases       int
	rate         int
}

// Prepare computes the filter for sampleRate, reusing f's storage. It does
// nothing when f already holds that rate.
func (f *Filter) Prepare(sampleRate int) {
	if f.rate == sampleRate && len(f.Coefficients) != 0 {
		return
	}
	phases := Rate / gcd(sampleRate, Rate)
	needed := phases * Taps
	if cap(f.Coefficients) < needed {
		f.Coefficients = make([]float64, needed)
	} else {
		f.Coefficients = f.Coefficients[:needed]
	}
	cutoff := float64(Rate) / float64(sampleRate)
	if cutoff > 1 {
		cutoff = 1
	}
	remainder := 0
	for phase := 0; phase < phases; phase++ {
		fraction := float64(remainder) / Rate
		var norm float64
		for tap := 0; tap < Taps; tap++ {
			offset := tap - Radius + 1
			distance := float64(offset) - fraction
			weight := cutoff * sinc(cutoff*distance)
			if math.Abs(distance) < Radius {
				weight *= 0.5 + 0.5*math.Cos(math.Pi*distance/Radius)
			} else {
				weight = 0
			}
			f.Coefficients[phase*Taps+tap] = weight
			norm += weight
		}
		if norm != 0 {
			for tap := 0; tap < Taps; tap++ {
				f.Coefficients[phase*Taps+tap] /= norm
			}
		}
		remainder = (remainder + sampleRate) % Rate
	}
	f.rate = sampleRate
	f.Phases = phases
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
