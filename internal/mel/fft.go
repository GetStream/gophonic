// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package mel

import (
	"math"
	"sync"
)

// float is the precision of a frontend: the turn detectors compute in
// float64, Whisper and Qwen3-ASR in float32.
type float interface{ float32 | float64 }

// tables holds the precomputed inputs of the 400-point STFT in one
// precision: the periodic Hann window and the DFT roots exp(-2πik/400).
type tables[T float] struct {
	hann, rootRe, rootIm [FFTSize]T
}

var (
	tables32 = newTables[float32]()
	tables64 = newTables[float64]()
	// fftOrder is the mixed-radix digit reversal of the 2x2x2x2x5x5
	// decimation-in-time transform: input n goes to fftOrder[n].
	fftOrder = func() (order [FFTSize]int) {
		radices := [...]int{2, 2, 2, 2, 5, 5}
		for n := range order {
			value := n
			var digits [len(radices)]int
			for i := len(radices) - 1; i >= 0; i-- {
				digits[i] = value % radices[i]
				value /= radices[i]
			}
			reversed, stride := 0, 1
			for i, radix := range radices {
				reversed += digits[i] * stride
				stride *= radix
			}
			order[n] = reversed
		}
		return order
	}()
)

func newTables[T float]() *tables[T] {
	t := new(tables[T])
	for n := range t.hann {
		t.hann[n] = T(0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/FFTSize))
	}
	for k := range t.rootRe {
		angle := 2 * math.Pi * float64(k) / FFTSize
		t.rootRe[k] = T(math.Cos(angle))
		t.rootIm[k] = T(-math.Sin(angle))
	}
	return t
}

// powerFrame writes the power spectrum of the Hann-windowed 400 samples
// frame to power[bin*stride], using re and im as scratch.
func powerFrame[T float](t *tables[T], frame []T, re, im *[FFTSize]T, power []T, stride int) {
	samples := (*[FFTSize]T)(frame) // one length check instead of 400
	for n, index := range &fftOrder {
		re[index] = samples[n] * t.hann[n]
		im[index] = 0
	}
	fft400(t, re, im)
	_ = power[(fftBins-1)*stride]
	for bin := 0; bin < fftBins; bin++ {
		r, i := re[bin], im[bin]
		power[bin*stride] = r*r + i*i
	}
}

// fft400 computes the unscaled, negative-exponent 400-point DFT in place of
// digit-reversed input. Its radices are 2, 2, 2, 2, 5, and 5, so every stage
// is an exact factor of 400 and the transform needs no scratch memory.
func fft400[T float](t *tables[T], re, im *[FFTSize]T) {
	fftRadix2(t, re, im, 1)
	fftRadix2(t, re, im, 2)
	fftRadix2(t, re, im, 4)
	fftRadix2(t, re, im, 8)
	fftRadix5(t, re, im, 16)
	fftRadix5(t, re, im, 80)
}

func fftRadix2[T float](t *tables[T], realPart, imaginaryPart *[FFTSize]T, previousLength int) {
	stageLength := previousLength * 2
	rootScale := FFTSize / stageLength
	for base := 0; base < FFTSize; base += stageLength {
		for offset := 0; offset < previousLength; offset++ {
			left := base + offset
			right := left + previousLength
			rightReal, rightImag := realPart[right], imaginaryPart[right]
			if offset != 0 {
				cos, sin := t.rootRe[offset*rootScale], t.rootIm[offset*rootScale]
				rightReal, rightImag = rightReal*cos-rightImag*sin, rightReal*sin+rightImag*cos
			}
			leftReal, leftImag := realPart[left], imaginaryPart[left]
			realPart[left], imaginaryPart[left] = leftReal+rightReal, leftImag+rightImag
			realPart[right], imaginaryPart[right] = leftReal-rightReal, leftImag-rightImag
		}
	}
}

func fftRadix5[T float](t *tables[T], realPart, imaginaryPart *[FFTSize]T, previousLength int) {
	var (
		cos72  = T(0.30901699437494745)
		cos144 = T(-0.80901699437494745)
		sin72  = T(0.95105651629515357)
		sin144 = T(0.58778525229247314)
	)
	stageLength := previousLength * 5
	rootScale := FFTSize / stageLength
	for base := 0; base < FFTSize; base += stageLength {
		for offset := 0; offset < previousLength; offset++ {
			index0 := base + offset
			index1 := index0 + previousLength
			index2 := index1 + previousLength
			index3 := index2 + previousLength
			index4 := index3 + previousLength
			x0Real, x0Imag := realPart[index0], imaginaryPart[index0]
			x1Real, x1Imag := realPart[index1], imaginaryPart[index1]
			x2Real, x2Imag := realPart[index2], imaginaryPart[index2]
			x3Real, x3Imag := realPart[index3], imaginaryPart[index3]
			x4Real, x4Imag := realPart[index4], imaginaryPart[index4]
			if offset != 0 {
				root := offset * rootScale
				cos, sin := t.rootRe[root], t.rootIm[root]
				x1Real, x1Imag = x1Real*cos-x1Imag*sin, x1Real*sin+x1Imag*cos
				root *= 2
				cos, sin = t.rootRe[root], t.rootIm[root]
				x2Real, x2Imag = x2Real*cos-x2Imag*sin, x2Real*sin+x2Imag*cos
				root += offset * rootScale
				cos, sin = t.rootRe[root], t.rootIm[root]
				x3Real, x3Imag = x3Real*cos-x3Imag*sin, x3Real*sin+x3Imag*cos
				root += offset * rootScale
				cos, sin = t.rootRe[root], t.rootIm[root]
				x4Real, x4Imag = x4Real*cos-x4Imag*sin, x4Real*sin+x4Imag*cos
			}
			sum14Real, sum14Imag := x1Real+x4Real, x1Imag+x4Imag
			sum23Real, sum23Imag := x2Real+x3Real, x2Imag+x3Imag
			diff14Real, diff14Imag := x1Real-x4Real, x1Imag-x4Imag
			diff23Real, diff23Imag := x2Real-x3Real, x2Imag-x3Imag
			base1Real := x0Real + cos72*sum14Real + cos144*sum23Real
			base1Imag := x0Imag + cos72*sum14Imag + cos144*sum23Imag
			base2Real := x0Real + cos144*sum14Real + cos72*sum23Real
			base2Imag := x0Imag + cos144*sum14Imag + cos72*sum23Imag
			uReal := sin72*diff14Real + sin144*diff23Real
			uImag := sin72*diff14Imag + sin144*diff23Imag
			vReal := sin144*diff14Real - sin72*diff23Real
			vImag := sin144*diff14Imag - sin72*diff23Imag
			realPart[index0], imaginaryPart[index0] = x0Real+sum14Real+sum23Real, x0Imag+sum14Imag+sum23Imag
			realPart[index1], imaginaryPart[index1] = base1Real+uImag, base1Imag-uReal
			realPart[index4], imaginaryPart[index4] = base1Real-uImag, base1Imag+uReal
			realPart[index2], imaginaryPart[index2] = base2Real+vImag, base2Imag-vReal
			realPart[index3], imaginaryPart[index3] = base2Real-vImag, base2Imag+vReal
		}
	}
}

// bank is a Slaney-normalized triangular mel filter bank over 0–8 kHz on the
// 201 STFT bins, as Whisper (librosa) and Hugging Face's
// mel_filter_bank(norm="slaney", mel_scale="slaney") build it: rows[mel][bin].
type bank[T float] struct{ rows [][fftBins]T }

var banks struct {
	sync.Mutex
	f32 map[int]*bank[float32]
	f64 map[int]*bank[float64]
}

// bank32 and bank64 return the cached bank with bins bands.
func bank32(bins int) *bank[float32] {
	banks.Lock()
	defer banks.Unlock()
	if banks.f32 == nil {
		banks.f32 = map[int]*bank[float32]{}
	}
	b := banks.f32[bins]
	if b == nil {
		b = newBank[float32](bins)
		banks.f32[bins] = b
	}
	return b
}

func bank64(bins int) *bank[float64] {
	banks.Lock()
	defer banks.Unlock()
	if banks.f64 == nil {
		banks.f64 = map[int]*bank[float64]{}
	}
	b := banks.f64[bins]
	if b == nil {
		b = newBank[float64](bins)
		banks.f64[bins] = b
	}
	return b
}

func newBank[T float](bins int) *bank[T] {
	b := &bank[T]{rows: make([][fftBins]T, bins)}
	melPoints := make([]float64, bins+2)
	hzPoints := make([]float64, bins+2)
	minMel, maxMel := hertzToMel(0), hertzToMel(8000)
	for i := range melPoints {
		melPoints[i] = minMel + (maxMel-minMel)*float64(i)/float64(len(melPoints)-1)
		hzPoints[i] = melToHertz(melPoints[i])
	}
	for mel := range bins {
		leftWidth := hzPoints[mel+1] - hzPoints[mel]
		rightWidth := hzPoints[mel+2] - hzPoints[mel+1]
		norm := 2 / (hzPoints[mel+2] - hzPoints[mel])
		for bin := 0; bin < fftBins; bin++ {
			frequency := float64(bin) * 8000 / (fftBins - 1)
			down := (frequency - hzPoints[mel]) / leftWidth
			up := (hzPoints[mel+2] - frequency) / rightWidth
			if value := math.Min(down, up); value > 0 {
				b.rows[mel][bin] = T(value * norm)
			}
		}
	}
	return b
}

// hertzToMel and melToHertz are the Slaney mel scale: linear below 1 kHz,
// logarithmic above.
func hertzToMel(frequency float64) float64 {
	mel := 3 * frequency / 200
	if frequency >= 1000 {
		mel = 15 + math.Log(frequency/1000)*(27/math.Log(6.4))
	}
	return mel
}

func melToHertz(mel float64) float64 {
	frequency := 200 * mel / 3
	if mel >= 15 {
		frequency = 1000 * math.Exp((math.Log(6.4)/27)*(mel-15))
	}
	return frequency
}
