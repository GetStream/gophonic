// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"math"
	"runtime"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

const (
	AudioFFNSize = AudioState * 4
)

var (
	errNilModel          = errors.New("whisper: nil model")
	errNilEncoderScratch = errors.New("whisper: nil or closed encoder workspace")
	errMelShape          = errors.New("whisper: mel input must contain 80x3000 values")
	errEncoderOutputSize = errors.New("whisper: encoder output buffer is too short")
	errEncoderWeights    = errors.New("whisper: model is missing or has invalid encoder weights")
	errEncoderGEMM       = errors.New("whisper: failed to initialize packed encoder projections")
)

// EncoderWorkspace owns temporary storage for one concurrent encoder call.
// Allocate one workspace per concurrent caller and reuse it between calls.
type EncoderWorkspace struct {
	conv1       []float32
	convColumns []float32
	normalized  []float32
	q           []float32
	k           []float32
	v           []float32
	feedForward []float32
	packedModel *Model
	packed      [AudioLayers]packedEncoderBlock
	conv1Weight *whispergemm.PackedB
	conv2Weight *whispergemm.PackedB
	attention   *audioAttention
	activation  encoderActivation
	gemm        *whispergemm.Executor
	closed      bool
}

type packedEncoderBlock struct {
	query, key, value, out *whispergemm.PackedB
	mlpIn, mlpOut          *whispergemm.PackedB
}

// NewEncoderWorkspace prepares a reusable encoder with at most eight workers,
// capped by GOMAXPROCS. The first EncodeInto packs the model's weights; repeated
// calls with the same model allocate no memory. Close releases its workers.
func NewEncoderWorkspace() *EncoderWorkspace {
	workers := min(runtime.GOMAXPROCS(0), 8)
	w, _ := NewEncoderWorkspaceWithWorkers(workers)
	return w
}

// NewEncoderWorkspaceWithWorkers creates an encoder workspace with a bounded
// pool of reusable GEMM workers. Workers includes the caller goroutine and must
// be between 1 and 64. Close the workspace to release its workers.
func NewEncoderWorkspaceWithWorkers(workers int) (*EncoderWorkspace, error) {
	gemm, err := whispergemm.NewExecutor(workers)
	if err != nil {
		return nil, err
	}
	w := &EncoderWorkspace{
		conv1:       make([]float32, MelFrames*AudioState),
		convColumns: make([]float32, AudioFrames*AudioState*3),
		normalized:  make([]float32, AudioFrames*AudioState),
		q:           make([]float32, AudioFrames*AudioState),
		k:           make([]float32, AudioFrames*AudioState),
		v:           make([]float32, AudioFrames*AudioState),
		feedForward: make([]float32, AudioFrames*AudioFFNSize),
		gemm:        gemm,
	}
	// All shapes are fixed positive tiny.en dimensions and cannot overflow.
	w.conv1Weight, _ = whispergemm.NewPackedB(MelBins*3, AudioState)
	w.conv2Weight, _ = whispergemm.NewPackedB(AudioState*3, AudioState)
	w.attention, err = newAudioAttention(AudioFrames, AudioState, AudioHeads, workers)
	if err != nil {
		w.Close()
		return nil, err
	}
	return w, nil
}

// Close drops the workspace buffers. A closed workspace cannot be reused.
func (w *EncoderWorkspace) Close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	w.conv1 = nil
	w.convColumns = nil
	w.normalized = nil
	w.q = nil
	w.k = nil
	w.v = nil
	w.feedForward = nil
	w.packedModel = nil
	w.packed = [AudioLayers]packedEncoderBlock{}
	w.conv1Weight = nil
	w.conv2Weight = nil
	w.attention = nil
	if w.gemm != nil {
		_ = w.gemm.Close()
		w.gemm = nil
	}
}

// EncodeInto runs Whisper tiny.en's audio encoder. mel is the flattened,
// channel-major [80,3000] log-mel tensor produced by the Whisper frontend. dst
// receives [1500,384] features in time-major order. The model and workspace
// must not be mutated or shared with another concurrent call, respectively.
func (m *Model) EncodeInto(mel, dst []float32, w *EncoderWorkspace) error {
	return m.encodeInto(mel, dst, w, nil)
}

func (m *Model) encodeInto(mel, dst []float32, w *EncoderWorkspace, trace func(stage int, values []float32)) error {
	if m == nil {
		return errNilModel
	}
	if w == nil || w.closed {
		return errNilEncoderScratch
	}
	if len(mel) != MelBins*MelFrames {
		return errMelShape
	}
	outLen := AudioFrames * AudioState
	if len(dst) < outLen {
		return errEncoderOutputSize
	}

	weights, ok := bindEncoderWeights(m)
	if !ok || !w.valid() {
		return errEncoderWeights
	}
	if err := w.preparePacked(m, weights); err != nil {
		return err
	}
	dst = dst[:outLen]

	// Whisper's audio stem is Conv1d -> exact GELU -> Conv1d (stride 2) ->
	// exact GELU -> fixed sinusoidal positions.
	lowerChannelMajor3(mel, w.convColumns, MelFrames, MelBins)
	if err := w.gemm.Mul(w.conv1Weight, w.conv1, AudioState, w.convColumns, MelBins*3, MelFrames); err != nil {
		return err
	}
	addRowBias(w.conv1, weights.conv1B, MelFrames, AudioState)
	if trace != nil {
		trace(0, w.conv1[:MelFrames*AudioState])
	}
	if err := w.activate(w.conv1, nil, MelFrames, AudioState); err != nil {
		return err
	}
	lowerTimeMajor3Stride2(w.conv1, w.convColumns, MelFrames, AudioFrames, AudioState)
	if err := w.gemm.Mul(w.conv2Weight, dst, AudioState, w.convColumns, AudioState*3, AudioFrames); err != nil {
		return err
	}
	addRowBias(dst, weights.conv2B, AudioFrames, AudioState)
	if trace != nil {
		trace(1, dst)
	}
	if err := w.activate(dst, nil, AudioFrames, AudioState); err != nil {
		return err
	}
	addPositionEmbedding(dst, weights.positions)

	for i := 0; i < AudioLayers; i++ {
		if err := encodeBlock(dst, weights.blocks[i], w.packed[i], w); err != nil {
			return err
		}
		if trace != nil {
			trace(i+2, dst)
		}
	}
	for t := 0; t < AudioFrames; t++ {
		start := t * AudioState
		row := dst[start : start+AudioState]
		layerNormRow(row, row, weights.finalNormW, weights.finalNormB)
	}
	if trace != nil {
		trace(AudioLayers+2, dst)
	}
	return nil
}

func encodeBlock(dst []float32, block encoderBlockWeights, packed packedEncoderBlock, w *EncoderWorkspace) error {
	for t := 0; t < AudioFrames; t++ {
		row := dst[t*AudioState : (t+1)*AudioState]
		layerNormRow(row, w.normalized[t*AudioState:(t+1)*AudioState], block.attnNormW, block.attnNormB)
	}
	if err := w.gemm.Mul(packed.query, w.q, AudioState, w.normalized, AudioState, AudioFrames); err != nil {
		return err
	}
	addRowBias(w.q, block.queryB, AudioFrames, AudioState)
	if err := w.gemm.Mul(packed.key, w.k, AudioState, w.normalized, AudioState, AudioFrames); err != nil {
		return err
	}
	if err := w.gemm.Mul(packed.value, w.v, AudioState, w.normalized, AudioState, AudioFrames); err != nil {
		return err
	}
	addRowBias(w.v, block.valueB, AudioFrames, AudioState)
	// Each query tile consumes its Q slice before the value product overwrites
	// it, avoiding another [1500,384] temporary.
	if err := w.attention.run(w.q, w.k, w.v, w.q, w.gemm); err != nil {
		return err
	}
	if err := w.gemm.Mul(packed.out, w.normalized, AudioState, w.q, AudioState, AudioFrames); err != nil {
		return err
	}
	addRowBias(w.normalized, block.outB, AudioFrames, AudioState)
	addInPlace(dst, w.normalized)

	for t := 0; t < AudioFrames; t++ {
		start := t * AudioState
		row := dst[start : start+AudioState]
		layerNormRow(row, w.normalized[start:start+AudioState], block.mlpNormW, block.mlpNormB)
	}
	if err := w.gemm.Mul(packed.mlpIn, w.feedForward, AudioFFNSize, w.normalized, AudioState, AudioFrames); err != nil {
		return err
	}
	if err := w.activate(w.feedForward, block.mlpInB, AudioFrames, AudioFFNSize); err != nil {
		return err
	}
	if err := w.gemm.Mul(packed.mlpOut, w.normalized, AudioState, w.feedForward, AudioFFNSize, AudioFrames); err != nil {
		return err
	}
	addRowBias(w.normalized, block.mlpOutB, AudioFrames, AudioState)
	addInPlace(dst, w.normalized)
	return nil
}

func (w *EncoderWorkspace) preparePacked(model *Model, weights encoderWeights) error {
	if w.packedModel == model {
		return nil
	}
	if w.gemm == nil {
		return errEncoderGEMM
	}
	if w.conv1Weight.Pack(weights.conv1W, MelBins*3, true) != nil ||
		w.conv2Weight.Pack(weights.conv2W, AudioState*3, true) != nil {
		return errEncoderGEMM
	}
	for i, block := range weights.blocks {
		packed := &w.packed[i]
		if packed.query == nil {
			var err error
			if packed.query, err = whispergemm.NewPackedB(AudioState, AudioState); err != nil {
				return errEncoderGEMM
			}
			if packed.key, err = whispergemm.NewPackedB(AudioState, AudioState); err != nil {
				return errEncoderGEMM
			}
			if packed.value, err = whispergemm.NewPackedB(AudioState, AudioState); err != nil {
				return errEncoderGEMM
			}
			if packed.out, err = whispergemm.NewPackedB(AudioState, AudioState); err != nil {
				return errEncoderGEMM
			}
			if packed.mlpIn, err = whispergemm.NewPackedB(AudioState, AudioFFNSize); err != nil {
				return errEncoderGEMM
			}
			if packed.mlpOut, err = whispergemm.NewPackedB(AudioFFNSize, AudioState); err != nil {
				return errEncoderGEMM
			}
		}
		if packed.query.Pack(block.queryW, AudioState, true) != nil ||
			packed.key.Pack(block.keyW, AudioState, true) != nil ||
			packed.value.Pack(block.valueW, AudioState, true) != nil ||
			packed.out.Pack(block.outW, AudioState, true) != nil ||
			packed.mlpIn.Pack(block.mlpInW, AudioState, true) != nil ||
			packed.mlpOut.Pack(block.mlpOutW, AudioFFNSize, true) != nil {
			return errEncoderGEMM
		}
	}
	w.packedModel = model
	return nil
}

func addRowBias(values, bias []float32, rows, width int) {
	if bias == nil {
		return
	}
	for row := 0; row < rows; row++ {
		start := row * width
		for col, value := range bias {
			values[start+col] += value
		}
	}
}

type encoderBlockWeights struct {
	queryW, queryB   []float32
	keyW             []float32
	valueW, valueB   []float32
	outW, outB       []float32
	attnNormW        []float32
	attnNormB        []float32
	mlpInW, mlpInB   []float32
	mlpOutW, mlpOutB []float32
	mlpNormW         []float32
	mlpNormB         []float32
}

type encoderWeights struct {
	conv1W, conv1B []float32
	conv2W, conv2B []float32
	positions      []float32
	blocks         [AudioLayers]encoderBlockWeights
	finalNormW     []float32
	finalNormB     []float32
}

// Keep full names static so binding weights on a warmed inference path cannot
// allocate while constructing per-layer map keys.
var encoderWeightNames = [AudioLayers]struct {
	queryW, queryB   string
	keyW             string
	valueW, valueB   string
	outW, outB       string
	attnNormW        string
	attnNormB        string
	mlpInW, mlpInB   string
	mlpOutW, mlpOutB string
	mlpNormW         string
	mlpNormB         string
}{
	{
		queryW: "encoder.blocks.0.attn.query.weight", queryB: "encoder.blocks.0.attn.query.bias",
		keyW:   "encoder.blocks.0.attn.key.weight",
		valueW: "encoder.blocks.0.attn.value.weight", valueB: "encoder.blocks.0.attn.value.bias",
		outW: "encoder.blocks.0.attn.out.weight", outB: "encoder.blocks.0.attn.out.bias",
		attnNormW: "encoder.blocks.0.attn_ln.weight", attnNormB: "encoder.blocks.0.attn_ln.bias",
		mlpInW: "encoder.blocks.0.mlp.0.weight", mlpInB: "encoder.blocks.0.mlp.0.bias",
		mlpOutW: "encoder.blocks.0.mlp.2.weight", mlpOutB: "encoder.blocks.0.mlp.2.bias",
		mlpNormW: "encoder.blocks.0.mlp_ln.weight", mlpNormB: "encoder.blocks.0.mlp_ln.bias",
	},
	{
		queryW: "encoder.blocks.1.attn.query.weight", queryB: "encoder.blocks.1.attn.query.bias",
		keyW:   "encoder.blocks.1.attn.key.weight",
		valueW: "encoder.blocks.1.attn.value.weight", valueB: "encoder.blocks.1.attn.value.bias",
		outW: "encoder.blocks.1.attn.out.weight", outB: "encoder.blocks.1.attn.out.bias",
		attnNormW: "encoder.blocks.1.attn_ln.weight", attnNormB: "encoder.blocks.1.attn_ln.bias",
		mlpInW: "encoder.blocks.1.mlp.0.weight", mlpInB: "encoder.blocks.1.mlp.0.bias",
		mlpOutW: "encoder.blocks.1.mlp.2.weight", mlpOutB: "encoder.blocks.1.mlp.2.bias",
		mlpNormW: "encoder.blocks.1.mlp_ln.weight", mlpNormB: "encoder.blocks.1.mlp_ln.bias",
	},
	{
		queryW: "encoder.blocks.2.attn.query.weight", queryB: "encoder.blocks.2.attn.query.bias",
		keyW:   "encoder.blocks.2.attn.key.weight",
		valueW: "encoder.blocks.2.attn.value.weight", valueB: "encoder.blocks.2.attn.value.bias",
		outW: "encoder.blocks.2.attn.out.weight", outB: "encoder.blocks.2.attn.out.bias",
		attnNormW: "encoder.blocks.2.attn_ln.weight", attnNormB: "encoder.blocks.2.attn_ln.bias",
		mlpInW: "encoder.blocks.2.mlp.0.weight", mlpInB: "encoder.blocks.2.mlp.0.bias",
		mlpOutW: "encoder.blocks.2.mlp.2.weight", mlpOutB: "encoder.blocks.2.mlp.2.bias",
		mlpNormW: "encoder.blocks.2.mlp_ln.weight", mlpNormB: "encoder.blocks.2.mlp_ln.bias",
	},
	{
		queryW: "encoder.blocks.3.attn.query.weight", queryB: "encoder.blocks.3.attn.query.bias",
		keyW:   "encoder.blocks.3.attn.key.weight",
		valueW: "encoder.blocks.3.attn.value.weight", valueB: "encoder.blocks.3.attn.value.bias",
		outW: "encoder.blocks.3.attn.out.weight", outB: "encoder.blocks.3.attn.out.bias",
		attnNormW: "encoder.blocks.3.attn_ln.weight", attnNormB: "encoder.blocks.3.attn_ln.bias",
		mlpInW: "encoder.blocks.3.mlp.0.weight", mlpInB: "encoder.blocks.3.mlp.0.bias",
		mlpOutW: "encoder.blocks.3.mlp.2.weight", mlpOutB: "encoder.blocks.3.mlp.2.bias",
		mlpNormW: "encoder.blocks.3.mlp_ln.weight", mlpNormB: "encoder.blocks.3.mlp_ln.bias",
	},
}

func bindEncoderWeights(m *Model) (encoderWeights, bool) {
	w := encoderWeights{
		conv1W:     m.tensor("encoder.conv1.weight"),
		conv1B:     m.tensor("encoder.conv1.bias"),
		conv2W:     m.tensor("encoder.conv2.weight"),
		conv2B:     m.tensor("encoder.conv2.bias"),
		positions:  m.tensor("encoder.positional_embedding"),
		finalNormW: m.tensor("encoder.ln_post.weight"),
		finalNormB: m.tensor("encoder.ln_post.bias"),
	}
	if len(w.conv1W) != AudioState*MelBins*3 || len(w.conv1B) != AudioState ||
		len(w.conv2W) != AudioState*AudioState*3 || len(w.conv2B) != AudioState ||
		len(w.positions) != AudioFrames*AudioState || len(w.finalNormW) != AudioState || len(w.finalNormB) != AudioState {
		return encoderWeights{}, false
	}
	for i, names := range encoderWeightNames {
		b := &w.blocks[i]
		b.queryW, b.queryB = m.tensor(names.queryW), m.tensor(names.queryB)
		b.keyW = m.tensor(names.keyW)
		b.valueW, b.valueB = m.tensor(names.valueW), m.tensor(names.valueB)
		b.outW, b.outB = m.tensor(names.outW), m.tensor(names.outB)
		b.attnNormW, b.attnNormB = m.tensor(names.attnNormW), m.tensor(names.attnNormB)
		b.mlpInW, b.mlpInB = m.tensor(names.mlpInW), m.tensor(names.mlpInB)
		b.mlpOutW, b.mlpOutB = m.tensor(names.mlpOutW), m.tensor(names.mlpOutB)
		b.mlpNormW, b.mlpNormB = m.tensor(names.mlpNormW), m.tensor(names.mlpNormB)
		if len(b.queryW) != AudioState*AudioState || len(b.queryB) != AudioState ||
			len(b.keyW) != AudioState*AudioState || len(b.valueW) != AudioState*AudioState || len(b.valueB) != AudioState ||
			len(b.outW) != AudioState*AudioState || len(b.outB) != AudioState ||
			len(b.attnNormW) != AudioState || len(b.attnNormB) != AudioState ||
			len(b.mlpInW) != AudioFFNSize*AudioState || len(b.mlpInB) != AudioFFNSize ||
			len(b.mlpOutW) != AudioState*AudioFFNSize || len(b.mlpOutB) != AudioState ||
			len(b.mlpNormW) != AudioState || len(b.mlpNormB) != AudioState {
			return encoderWeights{}, false
		}
	}
	return w, true
}

func (w *EncoderWorkspace) valid() bool {
	return len(w.conv1) >= MelFrames*AudioState &&
		len(w.convColumns) >= AudioFrames*AudioState*3 &&
		len(w.normalized) >= AudioFrames*AudioState &&
		len(w.q) >= AudioFrames*AudioState &&
		len(w.k) >= AudioFrames*AudioState && len(w.v) >= AudioFrames*AudioState &&
		len(w.feedForward) >= AudioFrames*AudioFFNSize &&
		w.conv1Weight != nil && w.conv2Weight != nil && w.attention != nil && w.gemm != nil
}

// conv1Audio consumes Whisper's channel-major mel [channel,time] tensor and
// writes time-major channels so the second convolution can use contiguous rows.
func conv1Audio(src, dst, weights, bias []float32) {
	conv1DChannelMajor(src, dst, weights, bias, MelFrames, MelBins, AudioState)
}

func conv1DChannelMajor(src, dst, weights, bias []float32, frames, inChannels, outChannels int) {
	for t := 0; t < frames; t++ {
		for oc := 0; oc < outChannels; oc++ {
			sum := bias[oc]
			row := oc * inChannels * 3
			for ic := 0; ic < inChannels; ic++ {
				in := ic * frames
				weight := row + ic*3
				if t > 0 {
					sum += src[in+t-1] * weights[weight]
				}
				sum += src[in+t] * weights[weight+1]
				if t+1 < frames {
					sum += src[in+t+1] * weights[weight+2]
				}
			}
			dst[t*outChannels+oc] = sum
		}
	}
}

// conv2Audio applies Whisper's stride-two padded convolution to time-major input.
func conv2Audio(src, dst, weights, bias []float32) {
	conv1DStride2TimeMajor(src, dst, weights, bias, MelFrames, AudioFrames, AudioState)
}

func conv1DStride2TimeMajor(src, dst, weights, bias []float32, inputFrames, outputFrames, channels int) {
	for t := 0; t < outputFrames; t++ {
		center := t * 2
		left := center - 1
		right := center + 1
		outRow := t * channels
		for oc := 0; oc < channels; oc++ {
			sum := bias[oc]
			row := oc * channels * 3
			for ic := 0; ic < channels; ic++ {
				weight := row + ic*3
				base := ic
				if left >= 0 {
					sum += src[left*channels+base] * weights[weight]
				}
				sum += src[center*channels+base] * weights[weight+1]
				if right < inputFrames {
					sum += src[right*channels+base] * weights[weight+2]
				}
			}
			dst[outRow+oc] = sum
		}
	}
}

func addPositionEmbedding(dst, positions []float32) {
	for i := range dst {
		dst[i] += positions[i]
	}
}

func applyGELU(values []float32) {
	for i, x := range values {
		values[i] = float32(0.5 * float64(x) * (1 + math.Erf(float64(x)*0.707106781186547524400844362104849039)))
	}
}

func layerNormRow(src, dst, gamma, beta []float32) {
	var sum float64
	for _, x := range src {
		sum += float64(x)
	}
	mean := sum / float64(len(src))
	var variance float64
	for _, x := range src {
		delta := float64(x) - mean
		variance += delta * delta
	}
	variance /= float64(len(src))
	invStd := 1 / math.Sqrt(variance+1e-5)
	for i, x := range src {
		normalized := (float64(x) - mean) * invStd
		dst[i] = float32(normalized*float64(gamma[i]) + float64(beta[i]))
	}
}

func linearRow(src, dst, weights, bias []float32) {
	for out := range dst {
		w := weights[out*len(src) : (out+1)*len(src)]
		var sum float32
		if bias != nil {
			sum = bias[out]
		}
		for in := range src {
			sum += src[in] * w[in]
		}
		dst[out] = sum
	}
}

// multiHeadAttention mutates q and k by the two equivalent n^-1/4 factors
// used in OpenAI Whisper's eager implementation, then computes full unmasked
// audio self-attention without materializing a heads x time x time matrix.
func multiHeadAttention(q, k, v, dst, scores []float32, rows, state, heads int) {
	headSize := state / heads
	scale := float32(math.Pow(float64(headSize), -0.25))
	for i := 0; i < rows*state; i++ {
		q[i] *= scale
		k[i] *= scale
	}
	for t := 0; t < rows; t++ {
		for head := 0; head < heads; head++ {
			qBase := t*state + head*headSize
			maxScore := float32(math.Inf(-1))
			for keyT := 0; keyT < rows; keyT++ {
				kBase := keyT*state + head*headSize
				var score float32
				for d := 0; d < headSize; d++ {
					score += q[qBase+d] * k[kBase+d]
				}
				scores[keyT] = score
				if score > maxScore {
					maxScore = score
				}
			}
			var total float32
			for keyT := 0; keyT < rows; keyT++ {
				p := float32(math.Exp(float64(scores[keyT] - maxScore)))
				scores[keyT] = p
				total += p
			}
			invTotal := 1 / total
			outBase := t*state + head*headSize
			for d := 0; d < headSize; d++ {
				var sum float32
				for keyT := 0; keyT < rows; keyT++ {
					sum += (scores[keyT] * invTotal) * v[keyT*state+head*headSize+d]
				}
				dst[outBase+d] = sum
			}
		}
	}
}

func addInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}
