// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"math"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

const (
	featureSampleRate = 16000
	featureSamples    = featureSampleRate * 30
	featureFFTSize    = 400
	featureHopLength  = 160
	featureFFTBins    = featureFFTSize/2 + 1
	featurePad        = featureFFTSize / 2
)

var (
	errNilFeatureWorkspace = errors.New("whisper: nil or closed feature workspace")
	errFeatureOutputSize   = errors.New("whisper: feature output must contain 80x3000 float32 values")
	errFeatureNonFinite    = errors.New("whisper: audio contains a NaN or infinite sample")
)

// FeatureWorkspace owns reusable scratch for one concurrent audio frontend.
// It must not be used by more than one call at a time.
type FeatureWorkspace struct {
	padded   []float32
	power    []float32
	lanes    []featureLane
	spectrum []float32 // SME path: [frame, re/im bin], then [frame, mel]
	op       featureOperation
	closed   bool
}

// NewFeatureWorkspace allocates scratch for allocation-free FeaturesInto calls.
// It uses about 4 MiB; allocate one workspace for each concurrent caller.
func NewFeatureWorkspace() *FeatureWorkspace {
	return &FeatureWorkspace{
		padded: make([]float32, featureSamples+2*featurePad),
		power:  make([]float32, featureFFTBins*MelFrames),
	}
}

// Close releases the workspace buffers. Calls after Close return an error.
// Close must not race with FeaturesInto.
func (w *FeatureWorkspace) Close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	w.padded = nil
	w.power = nil
}

// FeaturesInto computes OpenAI Whisper's 80-band, 3000-frame log-mel input.
// pcm must be mono float32 PCM sampled at 16 kHz. Samples after 30 seconds are
// ignored; shorter inputs are zero-padded on the right. No waveform
// normalization is applied. dst is channel-major [80,3000] and must have
// exactly MelBins*MelFrames elements. Once w is created, calls allocate no
// memory. The workspace must be private to this call lane.
//
// The transform follows OpenAI Whisper source commit
// 86098128c0b4f24f0e2aa2994de830614b474227: periodic Hann window, centered
// reflect padding, power spectrum, Whisper's 80-band Slaney mel bank,
// dropped final STFT frame, 1e-10 log floor, max-minus-8 dynamic floor, and
// (log10(mel)+4)/4 output scaling.
func FeaturesInto(pcm []float32, dst []float32, w *FeatureWorkspace) error {
	return featuresInto(pcm, dst, w, nil)
}

// maxFeatureShards bounds frontend parallelism; each shard owns FFT scratch.
const maxFeatureShards = 64

type featureLane struct {
	real, imag [featureFFTSize]float32
}

// featureOperation runs the frontend in three barrier-separated phases over
// disjoint ranges: STFT frames, mel rows with log10, then the dynamic floor.
// Each value is computed by the same operations as a serial pass, so results
// do not depend on the shard count.
type featureOperation struct {
	w             *FeatureWorkspace
	dst           []float32
	phase, shards int
	floor         float32
	maxes         [maxFeatureShards]float32
}

// Phases 0-2 are the FFT path; the SME path uses power, log, then floor.
const (
	featurePhaseFloor = 2
	featurePhasePower = 3
	featurePhaseLog   = 4
)

func (op *featureOperation) ApplyRows(first, last int) {
	w, dst := op.w, op.dst
	for shard := first; shard < last; shard++ {
		switch op.phase {
		case featurePhasePower:
			for frame := shard * MelFrames / op.shards; frame < (shard+1)*MelFrames/op.shards; frame++ {
				spectrum := w.spectrum[frame*featureSpectrumWidth : (frame+1)*featureSpectrumWidth]
				power := w.power[frame*featureFFTBins : (frame+1)*featureFFTBins]
				for bin := range power {
					re, im := spectrum[2*bin], spectrum[2*bin+1]
					power[bin] = re*re + im*im
				}
			}
		case featurePhaseLog:
			mel := w.spectrum[:MelFrames*MelBins]
			maxLog := float32(math.Inf(-1))
			for m := shard * MelBins / op.shards; m < (shard+1)*MelBins/op.shards; m++ {
				row := dst[m*MelFrames : (m+1)*MelFrames]
				for frame := range row {
					value := mel[frame*MelBins+m]
					if value < 1e-10 {
						value = 1e-10
					}
					logMel := float32(math.Log10(float64(value)))
					row[frame] = logMel
					maxLog = max(maxLog, logMel)
				}
			}
			op.maxes[shard] = maxLog
		case 0:
			lane := &w.lanes[shard]
			for frame := shard * MelFrames / op.shards; frame < (shard+1)*MelFrames/op.shards; frame++ {
				start := frame * featureHopLength
				for n, index := range featureFFTOrder {
					lane.real[index] = w.padded[start+n] * featureHann[n]
					lane.imag[index] = 0
				}
				fftFeature400(&lane.real, &lane.imag)
				for bin := 0; bin < featureFFTBins; bin++ {
					re, im := lane.real[bin], lane.imag[bin]
					w.power[bin*MelFrames+frame] = re*re + im*im
				}
			}
		case 1:
			// Match the [mel,frequency] @ [frequency,time] layout in Whisper.
			// Writing directly into dst lets it serve as mel scratch as well.
			maxLog := float32(math.Inf(-1))
			for mel := shard * MelBins / op.shards; mel < (shard+1)*MelBins/op.shards; mel++ {
				row := dst[mel*MelFrames : (mel+1)*MelFrames]
				clear(row)
				for bin, weight := range featureMelBank[mel] {
					if weight == 0 {
						continue
					}
					power := w.power[bin*MelFrames : (bin+1)*MelFrames]
					for frame, p := range power {
						row[frame] += p * weight
					}
				}
				for i, value := range row {
					if value < 1e-10 {
						value = 1e-10
					}
					logMel := float32(math.Log10(float64(value)))
					row[i] = logMel
					maxLog = max(maxLog, logMel)
				}
			}
			op.maxes[shard] = maxLog
			if op.shards == 1 {
				op.floor = maxLog - 8
			}
		case 2:
			for i := shard * len(dst) / op.shards; i < (shard+1)*len(dst)/op.shards; i++ {
				value := dst[i]
				if value < op.floor {
					value = op.floor
				}
				dst[i] = (value + 4) / 4
			}
		}
	}
}

func featuresInto(pcm []float32, dst []float32, w *FeatureWorkspace, executor *whispergemm.Executor) error {
	if w == nil || w.closed {
		return errNilFeatureWorkspace
	}
	if len(dst) != MelBins*MelFrames {
		return errFeatureOutputSize
	}
	if len(w.padded) != featureSamples+2*featurePad || len(w.power) != featureFFTBins*MelFrames {
		return errNilFeatureWorkspace
	}

	used := len(pcm)
	if used > featureSamples {
		used = featureSamples
	}
	for i := 0; i < used; i++ {
		x := pcm[i]
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return errFeatureNonFinite
		}
	}
	center := w.padded[featurePad : featurePad+featureSamples]
	copy(center[:used], pcm[:used])
	clear(center[used:])

	// torch.stft(center=true) reflect-pads by n_fft/2, excluding each edge
	// sample, before applying its periodic Hann window.
	for i := 0; i < featurePad; i++ {
		w.padded[i] = center[featurePad-i]
		w.padded[featurePad+featureSamples+i] = center[featureSamples-2-i]
	}

	if whispergemm.PackedVectorAccelerated() {
		return featuresGEMM(dst, w, executor)
	}
	shards := 1
	if executor != nil {
		shards = min(executor.Workers(), maxFeatureShards)
	}
	if len(w.lanes) < shards {
		w.lanes = make([]featureLane, shards) // first parallel call only
	}
	w.op = featureOperation{w: w, dst: dst, shards: shards}
	defer func() { w.op = featureOperation{} }()
	for phase := 0; phase < 3; phase++ {
		w.op.phase = phase
		if shards == 1 {
			w.op.ApplyRows(0, 1)
			continue
		}
		if err := executor.Rows(&w.op, shards, 1); err != nil {
			return err
		}
		if phase == 1 {
			// Every shard has written its log values and maximum.
			maxLog := float32(math.Inf(-1))
			for _, m := range w.op.maxes[:shards] {
				maxLog = max(maxLog, m)
			}
			w.op.floor = maxLog - 8
		}
	}
	return nil
}

var (
	featureHann     [featureFFTSize]float32
	featureRootRe   [featureFFTSize]float32
	featureRootIm   [featureFFTSize]float32
	featureFFTOrder [featureFFTSize]int
	featureMelBank  [MelBins][featureFFTBins]float32
)

func init() {
	for n := range featureHann {
		featureHann[n] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/featureFFTSize))
	}
	for k := range featureRootRe {
		angle := 2 * math.Pi * float64(k) / featureFFTSize
		featureRootRe[k] = float32(math.Cos(angle))
		featureRootIm[k] = float32(-math.Sin(angle))
	}
	buildFeatureFFTOrder()
	buildFeatureMelBank()
}

func buildFeatureFFTOrder() {
	// A 400-point mixed-radix decimation-in-time FFT: 2*2*2*2*5*5.
	radices := [...]int{2, 2, 2, 2, 5, 5}
	for n := range featureFFTOrder {
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
		featureFFTOrder[n] = reversed
	}
}

func buildFeatureMelBank() {
	melPoints := [MelBins + 2]float64{}
	hzPoints := [MelBins + 2]float64{}
	minMel, maxMel := featureHertzToMel(0), featureHertzToMel(8000)
	for i := range melPoints {
		melPoints[i] = minMel + (maxMel-minMel)*float64(i)/float64(len(melPoints)-1)
		hzPoints[i] = featureMelToHertz(melPoints[i])
	}
	for mel := 0; mel < MelBins; mel++ {
		leftWidth := hzPoints[mel+1] - hzPoints[mel]
		rightWidth := hzPoints[mel+2] - hzPoints[mel+1]
		norm := 2 / (hzPoints[mel+2] - hzPoints[mel])
		for bin := 0; bin < featureFFTBins; bin++ {
			frequency := float64(bin) * 8000 / (featureFFTBins - 1)
			down := (frequency - hzPoints[mel]) / leftWidth
			up := (hzPoints[mel+2] - frequency) / rightWidth
			weight := math.Min(down, up)
			if weight > 0 {
				featureMelBank[mel][bin] = float32(weight * norm)
			}
		}
	}
}

func featureHertzToMel(frequency float64) float64 {
	mel := 3 * frequency / 200
	if frequency >= 1000 {
		mel = 15 + math.Log(frequency/1000)*(27/math.Log(6.4))
	}
	return mel
}

func featureMelToHertz(mel float64) float64 {
	frequency := 200 * mel / 3
	if mel >= 15 {
		frequency = 1000 * math.Exp((math.Log(6.4)/27)*(mel-15))
	}
	return frequency
}

func fftFeature400(realPart, imaginaryPart *[featureFFTSize]float32) {
	fftFeatureRadix2(realPart, imaginaryPart, 1)
	fftFeatureRadix2(realPart, imaginaryPart, 2)
	fftFeatureRadix2(realPart, imaginaryPart, 4)
	fftFeatureRadix2(realPart, imaginaryPart, 8)
	fftFeatureRadix5(realPart, imaginaryPart, 16)
	fftFeatureRadix5(realPart, imaginaryPart, 80)
}

func fftFeatureRadix2(realPart, imaginaryPart *[featureFFTSize]float32, previousLength int) {
	stageLength := previousLength * 2
	rootScale := featureFFTSize / stageLength
	for base := 0; base < featureFFTSize; base += stageLength {
		for offset := 0; offset < previousLength; offset++ {
			left := base + offset
			right := left + previousLength
			rightReal, rightImag := realPart[right], imaginaryPart[right]
			if offset != 0 {
				cos, sin := featureRootRe[offset*rootScale], featureRootIm[offset*rootScale]
				rightReal, rightImag = rightReal*cos-rightImag*sin, rightReal*sin+rightImag*cos
			}
			leftReal, leftImag := realPart[left], imaginaryPart[left]
			realPart[left], imaginaryPart[left] = leftReal+rightReal, leftImag+rightImag
			realPart[right], imaginaryPart[right] = leftReal-rightReal, leftImag-rightImag
		}
	}
}

func fftFeatureRadix5(realPart, imaginaryPart *[featureFFTSize]float32, previousLength int) {
	const (
		cos72  float32 = 0.30901699437494745
		cos144 float32 = -0.80901699437494745
		sin72  float32 = 0.95105651629515357
		sin144 float32 = 0.58778525229247314
	)
	stageLength := previousLength * 5
	rootScale := featureFFTSize / stageLength
	for base := 0; base < featureFFTSize; base += stageLength {
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
				cos, sin := featureRootRe[root], featureRootIm[root]
				x1Real, x1Imag = x1Real*cos-x1Imag*sin, x1Real*sin+x1Imag*cos
				root *= 2
				cos, sin = featureRootRe[root], featureRootIm[root]
				x2Real, x2Imag = x2Real*cos-x2Imag*sin, x2Real*sin+x2Imag*cos
				root += offset * rootScale
				cos, sin = featureRootRe[root], featureRootIm[root]
				x3Real, x3Imag = x3Real*cos-x3Imag*sin, x3Real*sin+x3Imag*cos
				root += offset * rootScale
				cos, sin = featureRootRe[root], featureRootIm[root]
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
