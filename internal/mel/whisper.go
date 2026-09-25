// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package mel

import (
	"errors"
	"math"
	"sync"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

// The float32 frontends follow OpenAI Whisper source commit
// 86098128c0b4f24f0e2aa2994de830614b474227, which Hugging Face's
// WhisperFeatureExtractor reproduces: periodic Hann window, centered reflect
// padding, power spectrum, Slaney mel bank, dropped final STFT frame, 1e-10
// log floor, max-minus-8 dynamic floor, and (log10(mel)+4)/4 scaling. No
// waveform normalization is applied.

const (
	// WindowSamples and WindowFrames describe Whisper's fixed 30-second
	// window of 16 kHz audio.
	WindowSamples = 30 * 16000
	WindowFrames  = WindowSamples / hopLength

	// maxShards bounds frontend parallelism; each shard owns FFT scratch.
	maxShards = 64
	// spectrogramChunk is the frames transformed per pass of Spectrogram.
	spectrogramChunk = 64
	// spectrumWidth is one frame's interleaved real and imaginary bins.
	spectrumWidth = 2 * fftBins
)

var (
	ErrNegativeSamples = errors.New("mel: sample count must not be negative")
	ErrTooLarge        = errors.New("mel: input is too large")
	ErrTooShort        = errors.New("mel: input is shorter than half an STFT window")
)

// Window computes Whisper's [bins, 3000] log-mel input for the first 30
// seconds of mono 16 kHz PCM; shorter inputs are zero-padded on the right.
// Frames and mel rows can be sharded across a whispergemm.Executor without
// changing a bit, and on machines with a streaming matrix unit the STFT and
// mel projection run as matrix products. A Window belongs to one call lane.
type Window struct {
	bins     int
	bank     *bank[float32]
	padded   []float32
	power    []float32
	lanes    []lane
	spectrum []float32 // matrix path: [frame, re/im bin], then [frame, mel]
	op       windowOp
}

type lane struct{ real, imag [FFTSize]float32 }

// NewWindow allocates a window frontend with bins mel bands (about 4 MiB for
// 80 bands).
func NewWindow(bins int) *Window {
	return &Window{
		bins:   bins,
		bank:   bank32(bins),
		padded: make([]float32, WindowSamples+2*featurePad),
		power:  make([]float32, fftBins*WindowFrames),
	}
}

// windowOp runs the frontend in three barrier-separated phases over
// disjoint ranges: STFT frames, mel rows with log10, then the dynamic floor.
// Each value is computed by the same operations as a serial pass, so results
// do not depend on the shard count.
type windowOp struct {
	w             *Window
	dst           []float32
	phase, shards int
	floor         float32
	maxes         [maxShards]float32
}

// Phases 0-2 are the FFT path; the matrix path uses power, log, then floor.
const (
	phaseFloor = 2
	phasePower = 3
	phaseLog   = 4
)

func (op *windowOp) ApplyRows(first, last int) {
	w, dst, bins := op.w, op.dst, op.w.bins
	for shard := first; shard < last; shard++ {
		switch op.phase {
		case phasePower:
			for frame := shard * WindowFrames / op.shards; frame < (shard+1)*WindowFrames/op.shards; frame++ {
				spectrum := w.spectrum[frame*spectrumWidth : (frame+1)*spectrumWidth]
				power := w.power[frame*fftBins : (frame+1)*fftBins]
				for bin := range power {
					re, im := spectrum[2*bin], spectrum[2*bin+1]
					power[bin] = re*re + im*im
				}
			}
		case phaseLog:
			mel := w.spectrum[:WindowFrames*bins]
			maxLog := float32(math.Inf(-1))
			for m := shard * bins / op.shards; m < (shard+1)*bins/op.shards; m++ {
				row := dst[m*WindowFrames : (m+1)*WindowFrames]
				for frame := range row {
					value := mel[frame*bins+m]
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
			for frame := shard * WindowFrames / op.shards; frame < (shard+1)*WindowFrames/op.shards; frame++ {
				start := frame * hopLength
				powerFrame(tables32, w.padded[start:start+FFTSize], &lane.real, &lane.imag, w.power[frame:], WindowFrames)
			}
		case 1:
			// Match the [mel,frequency] @ [frequency,time] layout in Whisper.
			// Writing directly into dst lets it serve as mel scratch as well.
			maxLog := float32(math.Inf(-1))
			for mel := shard * bins / op.shards; mel < (shard+1)*bins/op.shards; mel++ {
				row := dst[mel*WindowFrames : (mel+1)*WindowFrames]
				clear(row)
				for bin, weight := range w.bank.rows[mel] {
					if weight == 0 {
						continue
					}
					power := w.power[bin*WindowFrames : (bin+1)*WindowFrames]
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

// Into writes the channel-major [bins, 3000] features of pcm to dst. A nil
// executor runs on the calling goroutine. Warm calls allocate nothing.
func (w *Window) Into(pcm []float32, dst []float32, executor *whispergemm.Executor) error {
	if len(dst) != w.bins*WindowFrames {
		return ErrFeatureBuffer
	}
	used := min(len(pcm), WindowSamples)
	for i := 0; i < used; i++ {
		x := pcm[i]
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return ErrNonFinite
		}
	}
	center := w.padded[featurePad : featurePad+WindowSamples]
	copy(center[:used], pcm[:used])
	clear(center[used:])
	reflectPad(w.padded, WindowSamples)

	if whispergemm.PackedVectorAccelerated() {
		return w.matrixInto(dst, executor)
	}
	shards := 1
	if executor != nil {
		shards = min(executor.Workers(), maxShards)
	}
	if len(w.lanes) < shards {
		w.lanes = make([]lane, shards) // first parallel call only
	}
	w.op = windowOp{w: w, dst: dst, shards: shards}
	defer func() { w.op = windowOp{} }()
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

// reflectPad fills the featurePad samples on each side of the centered
// signal of length n, as torch.stft(center=true) does: reflection excludes
// each edge sample.
func reflectPad[T float](padded []T, n int) {
	center := padded[featurePad : featurePad+n]
	for i := 0; i < featurePad; i++ {
		padded[i] = center[featurePad-i]
		padded[featurePad+n+i] = center[n-2-i]
	}
}

// On machines with a streaming matrix unit the STFT runs as one matrix
// product. Frame f of the reflect-padded signal starts at f*hop, so the frame
// matrix is the padded buffer itself read with a 160-sample row stride. It
// multiplies a 400x402 matrix holding the periodic Hann window times the real
// and imaginary DFT bases. The mel projection is a second product with the
// transposed filter bank, which sums the bins in the same increasing order
// as the FFT path.
var matrices struct {
	sync.Mutex
	dft  *whispergemm.PackedB
	melT map[int]*whispergemm.PackedB
}

func windowMatrices(bins int) (dft, melT *whispergemm.PackedB) {
	m := &matrices
	m.Lock()
	defer m.Unlock()
	if m.dft == nil {
		basis := make([]float32, FFTSize*spectrumWidth)
		for n := 0; n < FFTSize; n++ {
			hann := 0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/FFTSize)
			for bin := 0; bin < fftBins; bin++ {
				angle := 2 * math.Pi * float64(n*bin%FFTSize) / FFTSize
				basis[n*spectrumWidth+2*bin] = float32(hann * math.Cos(angle))
				basis[n*spectrumWidth+2*bin+1] = float32(-hann * math.Sin(angle))
			}
		}
		m.dft, _ = whispergemm.NewPackedB(FFTSize, spectrumWidth)
		_ = m.dft.Pack(basis, spectrumWidth, false)
		m.melT = map[int]*whispergemm.PackedB{}
	}
	if m.melT[bins] == nil {
		rows := make([]float32, bins*fftBins)
		for mel, row := range bank32(bins).rows {
			copy(rows[mel*fftBins:], row[:])
		}
		p, _ := whispergemm.NewPackedB(fftBins, bins)
		_ = p.Pack(rows, fftBins, true)
		m.melT[bins] = p
	}
	return m.dft, m.melT[bins]
}

func mul(executor *whispergemm.Executor, b *whispergemm.PackedB, dst []float32, dstStride int, a []float32, aStride, rows int) error {
	if executor != nil {
		return executor.Mul(b, dst, dstStride, a, aStride, rows)
	}
	return b.Mul(dst, dstStride, a, aStride, rows)
}

// matrixInto runs after Into has built the padded signal.
func (w *Window) matrixInto(dst []float32, executor *whispergemm.Executor) error {
	dft, melT := windowMatrices(w.bins)
	if len(w.spectrum) < WindowFrames*spectrumWidth {
		w.spectrum = make([]float32, WindowFrames*spectrumWidth) // first call only
	}
	if err := mul(executor, dft, w.spectrum, spectrumWidth, w.padded, hopLength, WindowFrames); err != nil {
		return err
	}
	shards := 1
	if executor != nil {
		shards = min(executor.Workers(), maxShards)
	}
	w.op = windowOp{w: w, dst: dst, shards: shards}
	defer func() { w.op = windowOp{} }()
	run := func(phase int) error {
		w.op.phase = phase
		if shards == 1 {
			w.op.ApplyRows(0, 1)
			return nil
		}
		return executor.Rows(&w.op, shards, 1)
	}
	// w.power holds the [frame, bin] power spectrum here.
	if err := run(phasePower); err != nil {
		return err
	}
	mel := w.spectrum[:WindowFrames*w.bins] // [frame, mel]
	if err := mul(executor, melT, mel, w.bins, w.power, fftBins, WindowFrames); err != nil {
		return err
	}
	if err := run(phaseLog); err != nil {
		return err
	}
	maxLog := float32(math.Inf(-1))
	for _, m := range w.op.maxes[:shards] {
		maxLog = max(maxLog, m)
	}
	w.op.floor = maxLog - 8
	return run(phaseFloor)
}

// Spectrogram computes the log-mel features of a whole signal: rightPad
// silent samples are appended before the STFT (Whisper's full-file frontend
// appends 30 seconds; Hugging Face's WhisperFeatureExtractor without padding,
// as Qwen3-ASR uses it, appends none), and the dynamic floor is taken over
// the complete spectrogram. A Spectrogram belongs to one call lane; it grows
// to fit longer inputs and keeps that capacity.
type Spectrogram struct {
	bins, rightPad int
	bank           *bank[float32]
	padded         []float32
	power          []float32
	real, imag     [FFTSize]float32
}

// NewSpectrogram creates a whole-signal frontend with bins mel bands.
func NewSpectrogram(bins, rightPad int) *Spectrogram {
	return &Spectrogram{
		bins:     bins,
		rightPad: rightPad,
		bank:     bank32(bins),
		padded:   make([]float32, rightPad+2*featurePad),
		power:    make([]float32, fftBins*spectrogramChunk),
	}
}

// Frames returns the number of feature frames for samples mono 16 kHz
// samples: floor((samples+rightPad)/160), as the final STFT frame is dropped.
func (s *Spectrogram) Frames(samples int) (int, error) {
	return SpectrogramFrames(samples, s.rightPad, s.bins)
}

// SpectrogramFrames is Frames for a frontend with the given padding and bands.
func SpectrogramFrames(samples, rightPad, bins int) (int, error) {
	if samples < 0 {
		return 0, ErrNegativeSamples
	}
	maxInt := int(^uint(0) >> 1)
	if samples > maxInt-(rightPad+2*featurePad) {
		return 0, ErrTooLarge
	}
	if samples+rightPad <= featurePad {
		return 0, ErrTooShort
	}
	frames := (samples + rightPad) / hopLength
	if frames > maxInt/bins {
		return 0, ErrTooLarge
	}
	return frames, nil
}

// Into writes the channel-major [bins, frames] features of pcm to dst, where
// frames is Frames(len(pcm)). Calls with inputs no longer than an earlier
// call allocate nothing.
func (s *Spectrogram) Into(pcm []float32, dst []float32) error {
	frames, err := s.Frames(len(pcm))
	if err != nil {
		return err
	}
	if len(dst) != s.bins*frames {
		return ErrFeatureBuffer
	}
	for _, x := range pcm {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return ErrNonFinite
		}
	}

	centerLength := len(pcm) + s.rightPad
	paddedLength := centerLength + 2*featurePad
	if cap(s.padded) < paddedLength {
		s.padded = make([]float32, paddedLength)
	} else {
		s.padded = s.padded[:paddedLength]
	}
	center := s.padded[featurePad : featurePad+centerLength]
	copy(center, pcm)
	clear(center[len(pcm):])
	reflectPad(s.padded, centerLength)

	maxLog := float32(math.Inf(-1))
	for firstFrame := 0; firstFrame < frames; {
		count := min(frames-firstFrame, spectrogramChunk)
		power := s.power[:fftBins*count]
		for localFrame := 0; localFrame < count; localFrame++ {
			start := (firstFrame + localFrame) * hopLength
			powerFrame(tables32, s.padded[start:start+FFTSize], &s.real, &s.imag, power[localFrame:], count)
		}
		for mel := 0; mel < s.bins; mel++ {
			row := dst[mel*frames+firstFrame : mel*frames+firstFrame+count]
			clear(row)
			for bin, weight := range s.bank.rows[mel] {
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
