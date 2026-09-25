// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package smartturn

import (
	"errors"
	"math"

	"github.com/GetStream/gophonic/internal/mel"
	"github.com/GetStream/gophonic/internal/vec"
)

var errNilWorkspace = errors.New("workspace is nil or closed")

// PredictInto extracts features from interleaved mono or stereo PCM and runs
// Smart Turn v3.2. The audio is resampled to 16 kHz when needed.
func (m *Model) PredictInto(pcm []float32, sampleRate, channels int, ws *Workspace) (Prediction, error) {
	if m == nil {
		return Prediction{}, errors.New("model is nil")
	}
	if ws == nil || ws.closed {
		return Prediction{}, errNilWorkspace
	}
	if err := ws.audio.Load(pcm, sampleRate, channels); err != nil {
		return Prediction{}, err
	}
	if err := ws.audio.FeaturesInto(ws.features); err != nil {
		return Prediction{}, err
	}
	return m.PredictFeaturesInto(ws.features, ws)
}

// PredictMono16kInto runs Smart Turn v3.2 on mono 16 kHz PCM.
func (m *Model) PredictMono16kInto(pcm []float32, ws *Workspace) (Prediction, error) {
	return m.PredictInto(pcm, 16000, 1, ws)
}

// PredictFeaturesInto runs inference on row-major [80,800] normalized log-mel
// features. This entry point is useful when an application owns preprocessing.
func (m *Model) PredictFeaturesInto(features []float32, ws *Workspace) (Prediction, error) {
	if m == nil {
		return Prediction{}, errors.New("model is nil")
	}
	if ws == nil || ws.closed {
		return Prediction{}, errNilWorkspace
	}
	if len(features) != melCount*frameCount {
		return Prediction{}, mel.ErrFeatureBuffer
	}
	m.encode(features, ws)
	return m.classify(ws.hiddenA, ws), nil
}

func (m *Model) encode(features []float32, ws *Workspace) {
	// Whisper's two convolutions use channels-first storage internally.
	for t := 0; t < frameCount; t++ {
		window := ws.conv1Windows[t*melCount*3 : (t+1)*melCount*3]
		for c := 0; c < melCount; c++ {
			base := c * 3
			for k := 0; k < 3; k++ {
				frame := t + k - 1
				if frame >= 0 && frame < frameCount {
					window[base+k] = features[c*frameCount+frame]
				} else {
					window[base+k] = 0
				}
			}
		}
	}
	ws.job = parallelJob{kind: workConv1, input: ws.conv1Windows, output: ws.conv1, weights: m.conv1W, bias: m.conv1B}
	ws.runRows(hiddenSize)

	for t := 0; t < sequenceLength; t++ {
		window := ws.conv2Windows[t*hiddenSize*3 : (t+1)*hiddenSize*3]
		center := t * 2
		for c := 0; c < hiddenSize; c++ {
			base := c * 3
			for k := 0; k < 3; k++ {
				frame := center + k - 1
				if frame >= 0 && frame < frameCount {
					window[base+k] = ws.conv1[c*frameCount+frame]
				} else {
					window[base+k] = 0
				}
			}
		}
	}
	ws.job = parallelJob{kind: workConv2, input: ws.conv2Windows, output: ws.hiddenA, weights: m.conv2W, bias: m.conv2B, positions: m.positions}
	ws.runRows(hiddenSize)

	for layerIndex := range m.layers {
		layer := &m.layers[layerIndex]
		// Whisper encoder layers are pre-normalized: normalize before each
		// sublayer, then add the sublayer result to its residual.
		linearQKVInto(ws, ws.hiddenA, sequenceLength, layer.q, layer.k, layer.v, ws.q, ws.k, ws.v, hiddenSize, layer.selfNormW, layer.selfNormB, 1e-5)
		ws.job = parallelJob{kind: workAttention, input: ws.q, output: ws.attentionOut, q: ws.q, k: ws.k, v: ws.v, scores: ws.attention}
		ws.runRows((sequenceLength / 4) * attentionHeads)
		linearInto(ws, ws.attentionOut, sequenceLength, layer.out, ws.hiddenB, hiddenSize, hiddenSize)
		addInPlace(ws.hiddenB, ws.hiddenA)

		layerNormInto(ws, ws.hiddenB, ws.normed, sequenceLength, hiddenSize, layer.finalNormW, layer.finalNormB, 1e-5)
		linearInto(ws, ws.normed, sequenceLength, layer.fc1, ws.ffn, hiddenSize, feedForwardSize)
		geluInPlace(ws.ffn)
		linearInto(ws, ws.ffn, sequenceLength, layer.fc2, ws.hiddenA, feedForwardSize, hiddenSize)
		addInPlace(ws.hiddenA, ws.hiddenB)
	}
	layerNormInPlace(ws, ws.hiddenA, sequenceLength, hiddenSize, m.encoderNormW, m.encoderNormB, 1e-5)
}

func (m *Model) classify(hidden []float32, ws *Workspace) Prediction {
	linearInto(ws, hidden, sequenceLength, m.pool1, ws.poolHidden, hiddenSize, 256)
	for i, x := range ws.poolHidden {
		ws.poolHidden[i] = vec.Tanh(x)
	}
	linearInto(ws, ws.poolHidden, sequenceLength, m.pool2, ws.poolScores, 256, 1)
	softmaxInPlace(ws.poolScores)
	clear(ws.pooled)
	for t, weight := range ws.poolScores {
		row := hidden[t*hiddenSize : (t+1)*hiddenSize]
		for i, x := range row {
			ws.pooled[i] += weight * x
		}
	}
	linearInto(ws, ws.pooled, 1, m.classifier1, ws.classHidden, hiddenSize, 256)
	layerNormInto(ws, ws.classHidden, ws.classHidden, 1, 256, m.classifierNormW, m.classifierNormB, 1e-5)
	geluInPlace(ws.classHidden)
	linearInto(ws, ws.classHidden, 1, m.classifier2, ws.classMid, 256, 64)
	geluInPlace(ws.classMid)
	linearInto(ws, ws.classMid, 1, m.classifier3, ws.classLogit, 64, 1)
	p := float32(1 / (1 + math.Exp(-float64(ws.classLogit[0]))))
	return Prediction{Probability: p, Complete: p > 0.5}
}

func addInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func geluInPlace(values []float32) {
	for i, x := range values {
		values[i] = gelu(x)
	}
}

func gelu(x float32) float32 {
	const invSqrt2 = 0.7071067811865475244
	return 0.5 * x * (1 + vec.Erf(x*invSqrt2))
}

func softmaxInPlace(values []float32) {
	maxValue := float32(math.Inf(-1))
	for _, x := range values {
		if x > maxValue {
			maxValue = x
		}
	}
	var sum float32
	for i, x := range values {
		v := vec.ExpNegative(x - maxValue)
		values[i] = v
		sum += v
	}
	inv := 1 / sum
	for i := range values {
		values[i] *= inv
	}
}

// FeaturesInto writes Smart Turn's normalized [80,800] Whisper log-mel input
// from mono 16 kHz PCM. Short audio is left-padded; long audio keeps its last
// eight seconds. ws is reusable across calls.
func FeaturesInto(pcm []float32, dst []float32, ws *Workspace) error {
	if ws == nil || ws.closed {
		return errNilWorkspace
	}
	if err := ws.audio.Load(pcm, 16000, 1); err != nil {
		return err
	}
	return ws.audio.FeaturesInto(dst)
}
