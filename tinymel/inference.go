// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package tinymel

import (
	"errors"
	"math"
)

var errTinyFeatureShape = errors.New("TinyMelNet feature input must contain exactly 80x800 float32 values")

// PredictInto extracts the shared 16 kHz log-mel features and runs TinyMelNet.
// Audio is resampled to 16 kHz when needed; one Workspace is reusable
// by one concurrent prediction lane.
func (m *Model) PredictInto(pcm []float32, sampleRate, channels int, ws *Workspace) (Prediction, error) {
	if m == nil {
		return Prediction{}, errors.New("TinyMelNet model is nil")
	}
	if ws == nil || ws.closed {
		return Prediction{}, errTinyModelClosed
	}
	if err := ws.audio.Load(pcm, sampleRate, channels); err != nil {
		return Prediction{}, err
	}
	if err := ws.computeFeaturesParallel(ws.features); err != nil {
		return Prediction{}, err
	}
	return m.PredictFeaturesInto(ws.features, ws)
}

// PredictMono16kInto runs TinyMelNet on mono 16 kHz PCM.
func (m *Model) PredictMono16kInto(pcm []float32, ws *Workspace) (Prediction, error) {
	return m.PredictInto(pcm, 16000, 1, ws)
}

// PredictFeaturesInto runs TinyMelNet on the normalized row-major [80,800]
// log-mel features produced by the shared Whisper feature extractor.
func (m *Model) PredictFeaturesInto(features []float32, ws *Workspace) (Prediction, error) {
	if m == nil {
		return Prediction{}, errors.New("TinyMelNet model is nil")
	}
	if ws == nil || ws.closed {
		return Prediction{}, errTinyModelClosed
	}
	if len(features) != tinyMelCount*tinyFrameCount {
		return Prediction{}, errTinyFeatureShape
	}
	return m.predictFeatures(features, ws), nil
}

func (m *Model) predictFeatures(features []float32, ws *Workspace) Prediction {
	length := m.encodeTinyStem(features, ws)
	for i := range m.blocks {
		length = m.encodeTinyBlock(i, length, ws)
	}

	ws.runGRU(m, ws.convA[:tinySequenceLength*tinyStemChannels])
	tinyAttentionPool(m, ws)
	tinyLayerNorm(ws.pooled, ws.headNormed, m.headNormW, m.headNormB)
	params := quantizeTiny(ws.headNormed, ws.quantized)
	tinyQuantizedMatMul(m.head1, ws.quantized, params, ws.headHidden)
	ws.runGELU(ws.headHidden)
	params = quantizeTiny(ws.headHidden, ws.quantized)
	tinyQuantizedMatMul(m.head2, ws.quantized, params, ws.logit[:])

	logit := ws.logit[0]
	var probability float64
	if logit >= 0 {
		probability = 1 / (1 + math.Exp(-float64(logit)))
	} else {
		exp := math.Exp(float64(logit))
		probability = exp / (1 + exp)
	}
	p := float32(probability)
	return Prediction{Probability: p, Complete: p > 0.57}
}

func (m *Model) encodeTinyStem(features []float32, ws *Workspace) int {
	params := quantizeTinyMelForInference(features, ws.quantized, ws.melScratch)
	length := ws.runConvGELU(m.stem, ws.quantized, params, ws.convA, tinyFrameCount)
	return length
}

func (m *Model) encodeTinyBlock(index, inputLength int, ws *Workspace) int {
	params := quantizeTiny(ws.convA[:inputLength*tinyStemChannels], ws.quantized)
	length := ws.runConv(m.blocks[index].depthwise, ws.quantized, params, ws.convB, inputLength)
	params = quantizeTiny(ws.convB[:length*tinyStemChannels], ws.quantized)
	length = ws.runConvGELU(m.blocks[index].pointwise, ws.quantized, params, ws.convA, length)
	return length
}

func runTinyConvScalar(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength int) int {
	pad := conv.kernel / 2
	outputLength := (inputLength+2*pad-conv.kernel)/conv.stride + 1
	inPerGroup := conv.inChannels / conv.groups
	outPerGroup := conv.outChannels / conv.groups
	scale := params.scale * conv.weightScale
	for t := 0; t < outputLength; t++ {
		for oc := 0; oc < conv.outChannels; oc++ {
			group := oc / outPerGroup
			var accumulator int32
			for icg := 0; icg < inPerGroup; icg++ {
				ic := group*inPerGroup + icg
				weightBase := (oc*inPerGroup + icg) * conv.kernel
				for k := 0; k < conv.kernel; k++ {
					it := t*conv.stride + k - pad
					if it < 0 || it >= inputLength {
						continue
					}
					x := int32(input[it*conv.inChannels+ic]) - int32(params.zero)
					w := int32(conv.weight[weightBase+k]) - int32(conv.weightZero)
					accumulator += x * w
				}
			}
			value := float32(accumulator) * scale
			output[t*conv.outChannels+oc] = value + conv.bias[oc]
		}
	}
	return outputLength
}

func tinyGELUInPlace(values []float32) {
	tinyGELUInPlaceRange(values, 0, len(values))
}

func tinyGELUInPlaceRange(values []float32, from, to int) {
	const invSqrt2 = 0.7071067811865475244
	if from < 0 {
		from = 0
	}
	if to > len(values) {
		to = len(values)
	}
	for i := from; i < to; i++ {
		value := values[i]
		values[i] = 0.5 * value * float32(1+math.Erf(float64(value*invSqrt2)))
	}
}

func tinyQuantizedMatMul(layer tinyQuantizedLinear, input []uint8, params tinyQuantParams, output []float32) {
	scale := params.scale * layer.weightScale
	for o := 0; o < layer.out; o++ {
		var accumulator int32
		for i := 0; i < layer.in; i++ {
			x := int32(input[i]) - int32(params.zero)
			w := int32(layer.weight[i*layer.out+o]) - int32(layer.weightZero)
			accumulator += x * w
		}
		value := float32(accumulator) * scale
		output[o] = value + layer.bias[o]
	}
}

func tinyAttentionPool(m *Model, ws *Workspace) {
	hidden := ws.gru.buffer()
	params := quantizeTiny(hidden, ws.quantized)
	const scale float32 = 0.0625
	for t := 0; t < tinySequenceLength; t++ {
		var accumulator int32
		row := ws.quantized[t*2*tinyGRUHidden : (t+1)*2*tinyGRUHidden]
		for i, value := range row {
			accumulator += (int32(value) - int32(params.zero)) * (int32(m.poolQuery[i]) - int32(m.poolQueryZero))
		}
		ws.poolScores[t] = float32(accumulator) * (params.scale * m.poolQueryScale) * scale
	}
	tinySoftmax(ws.poolScores)
	clear(ws.pooled)
	for t, weight := range ws.poolScores {
		row := hidden[t*2*tinyGRUHidden : (t+1)*2*tinyGRUHidden]
		for i, value := range row {
			ws.pooled[i] += weight * value
		}
	}
}

func tinySoftmax(values []float32) {
	maxValue := float32(math.Inf(-1))
	for _, value := range values {
		if value > maxValue {
			maxValue = value
		}
	}
	var sum float32
	for i, value := range values {
		exp := float32(math.Exp(float64(value - maxValue)))
		values[i] = exp
		sum += exp
	}
	inv := 1 / sum
	for i := range values {
		values[i] *= inv
	}
}

func tinyLayerNorm(input, output, gamma, beta []float32) {
	var sum float32
	for _, value := range input {
		sum += value
	}
	mean := sum / float32(len(input))
	var variance float32
	for _, value := range input {
		delta := value - mean
		variance += delta * delta
	}
	variance /= float32(len(input))
	inv := 1 / float32(math.Sqrt(float64(variance+1e-5)))
	for i, value := range input {
		output[i] = (value-mean)*inv*gamma[i] + beta[i]
	}
}
