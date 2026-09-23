// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor

import (
	"errors"
	"math"
)

const (
	fftSize        = 400
	fftBins        = fftSize/2 + 1
	hopLength      = 160
	featurePad     = fftSize / 2
	resampleRadius = 16
	resampleTaps   = 2 * resampleRadius
)

var (
	errInvalidAudio         = errors.New("audio must contain complete mono or stereo PCM frames")
	errInvalidRate          = errors.New("sample rate must be between 8 kHz and 96 kHz")
	errNonFiniteAudio       = errors.New("audio contains a NaN or infinite sample")
	errInvalidFeatureBuffer = errors.New("feature output buffer must contain exactly 80x800 float32 values")
)

type audioWorkspace struct {
	samples              []float32
	padded               []float64
	power                []float64
	mel                  []float64
	logMel               []float64
	fftReal              [fftSize]float64
	fftImag              [fftSize]float64
	resampleCoefficients []float64
	resampleRate         int
	resamplePhases       int
}

func newAudioWorkspace() audioWorkspace {
	return audioWorkspace{
		samples: make([]float32, maxWindowSamples),
		padded:  make([]float64, maxWindowSamples+2*featurePad),
		power:   make([]float64, fftBins*frameCount),
		mel:     make([]float64, melCount*frameCount),
		logMel:  make([]float64, melCount*frameCount),
	}
}

var (
	hannWindow [fftSize]float64
	fftRootCos [fftSize]float64
	fftRootSin [fftSize]float64
	fftOrder   [fftSize]int
	melBank    [melCount][fftBins]float64
	fftRadices = [...]int{2, 2, 2, 2, 5, 5}
)

func init() {
	for n := range hannWindow {
		hannWindow[n] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/fftSize)
	}
	for k := range fftRootCos {
		angle := 2 * math.Pi * float64(k) / fftSize
		fftRootCos[k] = math.Cos(angle)
		fftRootSin[k] = -math.Sin(angle)
	}
	buildFFTOrder()
	buildMelBank()
}

// buildFFTOrder computes the mixed-radix digit-reversal permutation for the
// 2x2x2x2x5x5 decimation-in-time transform.
func buildFFTOrder() {
	for n := range fftOrder {
		value := n
		var digits [6]int
		for i := len(fftRadices) - 1; i >= 0; i-- {
			radix := fftRadices[i]
			digits[i] = value % radix
			value /= radix
		}
		reversed, stride := 0, 1
		for i, radix := range fftRadices {
			reversed += digits[i] * stride
			stride *= radix
		}
		fftOrder[n] = reversed
	}
}

// ExtractWhisperFeatures16k writes Smart Turn's normalized [80,800] Whisper
// log-mel input from mono PCM. Short audio is left-padded; long audio keeps its
// last eight seconds. ws is reusable across calls.
func ExtractWhisperFeatures16k(pcm []float32, dst []float32, ws *Workspace) error {
	if ws == nil || ws.closed {
		return errNilWorkspace
	}
	if len(dst) != melCount*frameCount {
		return errInvalidFeatureBuffer
	}
	if err := prepareAudio(pcm, 16000, 1, ws); err != nil {
		return err
	}
	return computeFeatures16k(ws.audio.samples, dst, &ws.audio)
}

func prepareAudio(pcm []float32, sampleRate, channels int, ws *Workspace) error {
	if sampleRate < 8000 || sampleRate > 96000 {
		return errInvalidRate
	}
	if channels < 1 || channels > 2 || len(pcm) == 0 || len(pcm)%channels != 0 {
		return errInvalidAudio
	}
	frames := len(pcm) / channels
	fullOutput64 := (int64(frames)*16000 + int64(sampleRate)/2) / int64(sampleRate)
	if fullOutput64 < 1 {
		return errInvalidAudio
	}
	outCount := int(fullOutput64)
	if outCount > maxWindowSamples {
		outCount = maxWindowSamples
	}
	sourceFirst := 0
	if fullOutput64 > maxWindowSamples {
		if sampleRate == 16000 {
			sourceFirst = frames - outCount
		} else {
			outputStart := fullOutput64 - int64(outCount)
			firstBase := int(outputStart * int64(sampleRate) / 16000)
			sourceFirst = firstBase - resampleRadius
			if sourceFirst < 0 {
				sourceFirst = 0
			}
		}
	}
	for frame := sourceFirst; frame < frames; frame++ {
		for channel := 0; channel < channels; channel++ {
			x := pcm[frame*channels+channel]
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return errNonFiniteAudio
			}
		}
	}
	leftPad := maxWindowSamples - outCount
	clear(ws.audio.samples)
	if sampleRate == 16000 {
		first := frames - outCount
		for i := 0; i < outCount; i++ {
			ws.audio.samples[leftPad+i] = monoAt(pcm, first+i, channels)
		}
		return nil
	}
	if err := ws.audio.prepareResampler(sampleRate); err != nil {
		return err
	}
	outStart := fullOutput64 - int64(outCount)
	rem := int64(sampleRate)
	denom := int64(16000)
	coefficients := ws.audio.resampleCoefficients
	for i := 0; i < outCount; i++ {
		globalOut := outStart + int64(i)
		numerator := globalOut * int64(sampleRate)
		base := numerator / denom
		phase := int(globalOut % int64(ws.audio.resamplePhases))
		coeff := coefficients[phase*resampleTaps : (phase+1)*resampleTaps]
		var value float64
		for tap := 0; tap < resampleTaps; tap++ {
			sourceFrame := base + int64(tap-resampleRadius+1)
			if sourceFrame >= 0 && sourceFrame < int64(frames) {
				value += float64(monoAt(pcm, int(sourceFrame), channels)) * coeff[tap]
			}
		}
		ws.audio.samples[leftPad+i] = float32(value)
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

func (w *audioWorkspace) prepareResampler(sampleRate int) error {
	if w.resampleRate == sampleRate && len(w.resampleCoefficients) != 0 {
		return nil
	}
	g := gcd(sampleRate, 16000)
	phases := 16000 / g
	if phases > 16000 {
		return errInvalidRate
	}
	needed := phases * resampleTaps
	if cap(w.resampleCoefficients) < needed {
		w.resampleCoefficients = make([]float64, needed)
	} else {
		w.resampleCoefficients = w.resampleCoefficients[:needed]
	}
	cutoff := 16000.0 / float64(sampleRate)
	if cutoff > 1 {
		cutoff = 1
	}
	rem := 0
	for phase := 0; phase < phases; phase++ {
		fraction := float64(rem) / 16000
		var norm float64
		for tap := 0; tap < resampleTaps; tap++ {
			offset := tap - resampleRadius + 1
			distance := float64(offset) - fraction
			weight := cutoff * sinc(cutoff*distance)
			if abs(distance) < resampleRadius {
				weight *= 0.5 + 0.5*math.Cos(math.Pi*distance/resampleRadius)
			} else {
				weight = 0
			}
			w.resampleCoefficients[phase*resampleTaps+tap] = weight
			norm += weight
		}
		if norm != 0 {
			for tap := 0; tap < resampleTaps; tap++ {
				w.resampleCoefficients[phase*resampleTaps+tap] /= norm
			}
		}
		rem = (rem + sampleRate) % 16000
	}
	w.resampleRate = sampleRate
	w.resamplePhases = phases
	return nil
}

func computeFeatures16k(audio []float32, output []float32, w *audioWorkspace) error {
	if len(audio) != maxWindowSamples || len(output) != melCount*frameCount {
		return errInvalidFeatureBuffer
	}
	prepareFeatures16k(audio, w)
	// Whisper uses a centered, reflect-padded 400-point power STFT. Its
	// 801st frame is dropped, leaving 800 frames for this model.
	fftPowerFrameRange(w, 0, frameCount, &w.fftReal, &w.fftImag)
	melRowRange(w, 0, melCount)
	finishFeatures16k(output, w)
	return nil
}

// prepareFeatures16k normalizes the fixed window and fills reflect padding.
// The FFT and mel stages may read the resulting padded samples concurrently.
func prepareFeatures16k(audio []float32, w *audioWorkspace) {
	var mean float32
	for _, x := range audio {
		mean += x
	}
	mean /= float32(maxWindowSamples)
	var variance float32
	for _, x := range audio {
		d := x - mean
		variance += d * d
	}
	variance /= float32(maxWindowSamples)
	denom := float32(math.Sqrt(float64(variance + 1e-7)))
	for i, x := range audio {
		value := (x - mean) / denom
		audio[i] = value
		w.padded[featurePad+i] = float64(value)
	}

	for i := 0; i < featurePad; i++ {
		w.padded[i] = float64(audio[featurePad-i])
		w.padded[featurePad+maxWindowSamples+i] = float64(audio[maxWindowSamples-2-i])
	}
}

// finishFeatures16k applies Whisper's log, dynamic range floor, and output
// normalization after all mel rows have been written.
func finishFeatures16k(output []float32, w *audioWorkspace) {
	maxLog := logMelRowRange(w, 0, melCount)
	writeFeatures16k(output, w, maxLog)
}

// logMelRowRange writes independent mel rows and returns their maximum.
// Reducing those maxima after a worker barrier preserves the serial result.
func logMelRowRange(w *audioWorkspace, from, to int) float64 {
	maxLog := math.Inf(-1)
	for i := from * frameCount; i < to*frameCount; i++ {
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

func writeFeatures16k(output []float32, w *audioWorkspace, maxLog float64) {
	floor := maxLog - 8
	for i, value := range w.logMel {
		if value < floor {
			value = floor
		}
		output[i] = float32((value + 4) * 0.25)
	}
}

// fftPowerFrameRange writes distinct frame columns of the frequency-major
// power buffer. Callers may run disjoint ranges concurrently with separate
// real/imaginary scratch arrays after the padded input has been prepared.
func fftPowerFrameRange(w *audioWorkspace, from, to int, realPart, imaginaryPart *[fftSize]float64) {
	for frame := from; frame < to; frame++ {
		start := frame * hopLength
		for n, index := range fftOrder {
			realPart[index] = w.padded[start+n] * hannWindow[n]
			imaginaryPart[index] = 0
		}
		fft400(realPart, imaginaryPart)
		for bin := 0; bin < fftBins; bin++ {
			real, imaginary := realPart[bin], imaginaryPart[bin]
			w.power[bin*frameCount+frame] = real*real + imaginary*imaginary
		}
	}
}

// melRowRange writes independent mel-band rows. Each row retains the same
// bin summation order as the serial Whisper reference.
func melRowRange(w *audioWorkspace, from, to int) {
	for mel := from; mel < to; mel++ {
		row := w.mel[mel*frameCount : (mel+1)*frameCount]
		clear(row)
		for bin, weight := range melBank[mel] {
			if weight == 0 {
				continue
			}
			power := w.power[bin*frameCount : (bin+1)*frameCount]
			for frame, p := range power {
				row[frame] += p * weight
			}
		}
	}
}

// fft400 computes the unscaled, negative-exponent 400-point DFT in place.
// Its radices are 2, 2, 2, 2, 5, and 5, so every stage is an exact factor of
// 400 and the transform costs O(N log N) without allocating scratch memory.
func fft400(realPart, imaginaryPart *[fftSize]float64) {
	fftRadix2(realPart, imaginaryPart, 1)
	fftRadix2(realPart, imaginaryPart, 2)
	fftRadix2(realPart, imaginaryPart, 4)
	fftRadix2(realPart, imaginaryPart, 8)
	fftRadix5(realPart, imaginaryPart, 16)
	fftRadix5(realPart, imaginaryPart, 80)
}

func fftRadix2(realPart, imaginaryPart *[fftSize]float64, previousLength int) {
	stageLength := previousLength * 2
	rootScale := fftSize / stageLength
	for base := 0; base < fftSize; base += stageLength {
		for offset := 0; offset < previousLength; offset++ {
			left := base + offset
			right := left + previousLength
			rightReal, rightImag := realPart[right], imaginaryPart[right]
			if offset != 0 {
				cos, sin := fftRootCos[offset*rootScale], fftRootSin[offset*rootScale]
				rightReal, rightImag = rightReal*cos-rightImag*sin, rightReal*sin+rightImag*cos
			}
			leftReal, leftImag := realPart[left], imaginaryPart[left]
			realPart[left], imaginaryPart[left] = leftReal+rightReal, leftImag+rightImag
			realPart[right], imaginaryPart[right] = leftReal-rightReal, leftImag-rightImag
		}
	}
}

func fftRadix5(realPart, imaginaryPart *[fftSize]float64, previousLength int) {
	const (
		cos72  = 0.30901699437494745
		cos144 = -0.80901699437494745
		sin72  = 0.95105651629515357
		sin144 = 0.58778525229247314
	)
	stageLength := previousLength * 5
	rootScale := fftSize / stageLength
	for base := 0; base < fftSize; base += stageLength {
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
				cos, sin := fftRootCos[root], fftRootSin[root]
				x1Real, x1Imag = x1Real*cos-x1Imag*sin, x1Real*sin+x1Imag*cos
				root *= 2
				cos, sin = fftRootCos[root], fftRootSin[root]
				x2Real, x2Imag = x2Real*cos-x2Imag*sin, x2Real*sin+x2Imag*cos
				root += offset * rootScale
				cos, sin = fftRootCos[root], fftRootSin[root]
				x3Real, x3Imag = x3Real*cos-x3Imag*sin, x3Real*sin+x3Imag*cos
				root += offset * rootScale
				cos, sin = fftRootCos[root], fftRootSin[root]
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

func buildMelBank() {
	melPoints := make([]float64, melCount+2)
	hzPoints := make([]float64, melCount+2)
	minMel, maxMel := hertzToMel(0), hertzToMel(8000)
	for i := range melPoints {
		melPoints[i] = minMel + (maxMel-minMel)*float64(i)/float64(len(melPoints)-1)
		hzPoints[i] = melToHertz(melPoints[i])
	}
	for mel := 0; mel < melCount; mel++ {
		leftWidth := hzPoints[mel+1] - hzPoints[mel]
		rightWidth := hzPoints[mel+2] - hzPoints[mel+1]
		norm := 2 / (hzPoints[mel+2] - hzPoints[mel])
		for bin := 0; bin < fftBins; bin++ {
			frequency := float64(bin) * 8000 / (fftBins - 1)
			down := (frequency - hzPoints[mel]) / leftWidth
			up := (hzPoints[mel+2] - frequency) / rightWidth
			value := math.Min(down, up)
			if value > 0 {
				melBank[mel][bin] = value * norm
			}
		}
	}
}

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

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
