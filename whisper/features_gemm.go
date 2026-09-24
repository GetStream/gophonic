// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"sync"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

// On machines with a streaming matrix unit the STFT runs as one matrix
// product. Frame f of the reflect-padded signal starts at f*hop, so the frame
// matrix is the padded buffer itself read with a 160-sample row stride. It
// multiplies a 400x402 matrix holding the periodic Hann window times the real
// and imaginary DFT bases. The mel projection is a second product with the
// transposed filter bank, which sums the bins in the same increasing order
// as the FFT path.
const featureSpectrumWidth = 2 * featureFFTBins

var featureMatrices struct {
	once       sync.Once
	dft, melT  *whispergemm.PackedB
	melBankRow []float32
}

func featureGEMMMatrices() (dft, melT *whispergemm.PackedB) {
	m := &featureMatrices
	m.once.Do(func() {
		basis := make([]float32, featureFFTSize*featureSpectrumWidth)
		for n := 0; n < featureFFTSize; n++ {
			hann := 0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/featureFFTSize)
			for bin := 0; bin < featureFFTBins; bin++ {
				angle := 2 * math.Pi * float64(n*bin%featureFFTSize) / featureFFTSize
				basis[n*featureSpectrumWidth+2*bin] = float32(hann * math.Cos(angle))
				basis[n*featureSpectrumWidth+2*bin+1] = float32(-hann * math.Sin(angle))
			}
		}
		m.dft, _ = whispergemm.NewPackedB(featureFFTSize, featureSpectrumWidth)
		_ = m.dft.Pack(basis, featureSpectrumWidth, false)
		m.melBankRow = make([]float32, MelBins*featureFFTBins)
		for mel := range featureMelBank {
			copy(m.melBankRow[mel*featureFFTBins:], featureMelBank[mel][:])
		}
		m.melT, _ = whispergemm.NewPackedB(featureFFTBins, MelBins)
		_ = m.melT.Pack(m.melBankRow, featureFFTBins, true)
	})
	return m.dft, m.melT
}

func featureMul(executor *whispergemm.Executor, b *whispergemm.PackedB, dst []float32, dstStride int, a []float32, aStride, rows int) error {
	if executor != nil {
		return executor.Mul(b, dst, dstStride, a, aStride, rows)
	}
	return b.Mul(dst, dstStride, a, aStride, rows)
}

// featuresGEMM runs after featuresInto has built the padded signal.
func featuresGEMM(dst []float32, w *FeatureWorkspace, executor *whispergemm.Executor) error {
	dft, melT := featureGEMMMatrices()
	if len(w.spectrum) < MelFrames*featureSpectrumWidth {
		w.spectrum = make([]float32, MelFrames*featureSpectrumWidth) // first call only
	}
	if err := featureMul(executor, dft, w.spectrum, featureSpectrumWidth, w.padded, featureHopLength, MelFrames); err != nil {
		return err
	}
	shards := 1
	if executor != nil {
		shards = min(executor.Workers(), maxFeatureShards)
	}
	w.op = featureOperation{w: w, dst: dst, shards: shards}
	defer func() { w.op = featureOperation{} }()
	run := func(phase int) error {
		w.op.phase = phase
		if shards == 1 {
			w.op.ApplyRows(0, 1)
			return nil
		}
		return executor.Rows(&w.op, shards, 1)
	}
	// w.power holds the [frame, bin] power spectrum here.
	if err := run(featurePhasePower); err != nil {
		return err
	}
	mel := w.spectrum[:MelFrames*MelBins] // [frame, mel]
	if err := featureMul(executor, melT, mel, MelBins, w.power, featureFFTBins, MelFrames); err != nil {
		return err
	}
	if err := run(featurePhaseLog); err != nil {
		return err
	}
	maxLog := float32(math.Inf(-1))
	for _, m := range w.op.maxes[:shards] {
		maxLog = max(maxLog, m)
	}
	w.op.floor = maxLog - 8
	return run(featurePhaseFloor)
}
