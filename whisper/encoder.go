// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"fmt"
	"math"
	"runtime"

	"github.com/GetStream/gophonic/internal/nn"
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
	convColumns []float32 // time-major mel with a zero row at each end
	conv1Pad    []float32 // conv1 output after one zero row of left padding
	normalized  []float32
	q           []float32
	k           []float32
	v           []float32
	feedForward []float32
	dims        Dims
	packedModel *Model
	weights     encoderWeights
	packed      []packedEncoderBlock
	conv1Weight *whispergemm.PackedB
	conv2Weight *whispergemm.PackedB
	attention   *audioAttention
	activation  encoderActivation
	rowOp       encoderRows
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
	return newEncoderWorkspace(TinyENDims, workers)
}

// newEncoderWorkspace sizes the scratch for one model's dimensions.
func newEncoderWorkspace(d Dims, workers int) (*EncoderWorkspace, error) {
	if !d.valid() {
		return nil, errEncoderWeights
	}
	gemm, err := whispergemm.NewExecutor(workers)
	if err != nil {
		return nil, err
	}
	state := d.AudioState
	w := &EncoderWorkspace{
		dims:        d,
		conv1Pad:    make([]float32, (MelFrames+1)*state),
		convColumns: make([]float32, (MelFrames+2)*MelBins),
		normalized:  make([]float32, AudioFrames*state),
		q:           make([]float32, AudioFrames*state),
		k:           make([]float32, AudioFrames*state),
		v:           make([]float32, AudioFrames*state),
		feedForward: make([]float32, AudioFrames*4*state),
		packed:      make([]packedEncoderBlock, d.AudioLayers),
		gemm:        gemm,
	}
	// Validated dimensions are small and positive, so these cannot fail.
	w.conv1 = w.conv1Pad[state:]
	w.conv1Weight, _ = whispergemm.NewPackedB(MelBins*3, state)
	w.conv2Weight, _ = whispergemm.NewPackedB(state*3, state)
	w.attention, err = newAudioAttention(AudioFrames, state, d.AudioHeads, workers)
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
	w.conv1Pad = nil
	w.convColumns = nil
	w.normalized = nil
	w.q = nil
	w.k = nil
	w.v = nil
	w.feedForward = nil
	w.packedModel = nil
	w.packed = nil
	w.weights = encoderWeights{}
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
	if w.dims != m.dims || !w.valid() {
		return errEncoderWeights
	}
	d := w.dims
	state := d.AudioState
	outLen := AudioFrames * state
	if len(dst) < outLen {
		return errEncoderOutputSize
	}
	if w.packedModel != m {
		weights, ok := bindEncoderWeights(m)
		if !ok {
			return errEncoderWeights
		}
		w.weights = weights
		if err := w.preparePacked(m, weights); err != nil {
			return err
		}
	}
	weights := w.weights
	dst = dst[:outLen]

	// Whisper's audio stem is Conv1d -> exact GELU -> Conv1d (stride 2) ->
	// exact GELU -> fixed sinusoidal positions.
	// Both stem convolutions are single GEMMs over overlapping input rows:
	// row t of the time-major mel starting at padded row t-1 holds the three
	// taps [x(t-1), x(t), x(t+1)] contiguously, as does row 2t-1 of conv1's
	// output for the stride-two conv2. Weights are packed tap-major to match.
	melT := w.convColumns[:(MelFrames+2)*MelBins]
	clear(melT[:MelBins])
	clear(melT[(MelFrames+1)*MelBins:])
	if err := w.rows(encoderRows{kind: rowsTransposeMel, src: mel, dst: melT[MelBins:], frames: MelFrames, width: MelBins}, MelFrames); err != nil {
		return err
	}
	if err := w.gemm.Mul(w.conv1Weight, w.conv1, state, melT, MelBins, MelFrames); err != nil {
		return err
	}
	if trace != nil {
		nn.AddRowBias(w.conv1, weights.conv1B, MelFrames, state)
		trace(0, w.conv1[:MelFrames*state])
		if err := w.activate(w.conv1, nil, MelFrames, state); err != nil {
			return err
		}
	} else if err := w.activate(w.conv1, weights.conv1B, MelFrames, state); err != nil {
		return err
	}
	if err := w.gemm.Mul(w.conv2Weight, dst, state, w.conv1Pad, 2*state, AudioFrames); err != nil {
		return err
	}
	if trace != nil {
		nn.AddRowBias(dst, weights.conv2B, AudioFrames, state)
		trace(1, dst)
		if err := w.activate(dst, nil, AudioFrames, state); err != nil {
			return err
		}
	} else if err := w.activate(dst, weights.conv2B, AudioFrames, state); err != nil {
		return err
	}
	if err := w.rows(encoderRows{kind: rowsPosition, dst: dst, src: weights.positions, width: state}, AudioFrames); err != nil {
		return err
	}
	if err := w.rows(encoderRows{kind: rowsNorm, dst: dst, out: w.normalized, normW: weights.blocks[0].attnNormW, normB: weights.blocks[0].attnNormB, width: state}, AudioFrames); err != nil {
		return err
	}

	for i := 0; i < d.AudioLayers; i++ {
		// Each block ends by normalizing its output with the next block's
		// attention LayerNorm, or with the final encoder LayerNorm.
		nextW, nextB := weights.finalNormW, weights.finalNormB
		if i+1 < d.AudioLayers {
			nextW, nextB = weights.blocks[i+1].attnNormW, weights.blocks[i+1].attnNormB
		}
		if err := encodeBlock(dst, weights.blocks[i], w.packed[i], w, nextW, nextB, trace != nil); err != nil {
			return err
		}
		if trace != nil {
			trace(i+2, dst)
		}
	}
	// The last block left the final LayerNorm output in w.normalized.
	copy(dst, w.normalized[:len(dst)])
	if trace != nil {
		trace(d.AudioLayers+2, dst)
	}
	return nil
}

// encodeBlock expects w.normalized to hold this block's attention LayerNorm of
// dst. It leaves LayerNorm(dst; nextW, nextB) there for the following stage.
// With unfused set, the trace path also needs the raw block output, which dst
// always holds.
func encodeBlock(dst []float32, block encoderBlockWeights, packed packedEncoderBlock, w *EncoderWorkspace, nextW, nextB []float32, unfused bool) error {
	_ = unfused
	state, ffn := w.dims.AudioState, 4*w.dims.AudioState
	if err := w.gemm.Mul(packed.query, w.q, state, w.normalized, state, AudioFrames); err != nil {
		return err
	}
	if err := w.gemm.Mul(packed.key, w.k, state, w.normalized, state, AudioFrames); err != nil {
		return err
	}
	if err := w.gemm.Mul(packed.value, w.v, state, w.normalized, state, AudioFrames); err != nil {
		return err
	}
	// Q and V biases are added inside the attention preparation pass. Each
	// query tile consumes its Q slice before the value product overwrites it.
	if err := w.attention.runBiased(w.q, w.k, w.v, w.q, block.queryB, block.valueB, w.gemm); err != nil {
		return err
	}
	if err := w.gemm.Mul(packed.out, w.normalized, state, w.q, state, AudioFrames); err != nil {
		return err
	}
	if err := w.rows(encoderRows{kind: rowsResidual, dst: dst, src: w.normalized, bias: block.outB, out: w.normalized,
		normW: block.mlpNormW, normB: block.mlpNormB, width: state}, AudioFrames); err != nil {
		return err
	}
	if err := w.gemm.Mul(packed.mlpIn, w.feedForward, ffn, w.normalized, state, AudioFrames); err != nil {
		return err
	}
	if err := w.activate(w.feedForward, block.mlpInB, AudioFrames, ffn); err != nil {
		return err
	}
	if err := w.gemm.Mul(packed.mlpOut, w.normalized, state, w.feedForward, ffn, AudioFrames); err != nil {
		return err
	}
	return w.rows(encoderRows{kind: rowsResidual, dst: dst, src: w.normalized, bias: block.mlpOutB, out: w.normalized,
		normW: nextW, normB: nextB, width: state}, AudioFrames)
}

func (w *EncoderWorkspace) preparePacked(model *Model, weights encoderWeights) error {
	state, ffn := w.dims.AudioState, 4*w.dims.AudioState
	if w.gemm == nil {
		return errEncoderGEMM
	}
	if w.conv1Weight.Pack(tapMajor(weights.conv1W, state, MelBins), MelBins*3, true) != nil ||
		w.conv2Weight.Pack(tapMajor(weights.conv2W, state, state), state*3, true) != nil {
		return errEncoderGEMM
	}
	for i, block := range weights.blocks {
		packed := &w.packed[i]
		if packed.query == nil {
			var err error
			if packed.query, err = whispergemm.NewPackedB(state, state); err != nil {
				return errEncoderGEMM
			}
			if packed.key, err = whispergemm.NewPackedB(state, state); err != nil {
				return errEncoderGEMM
			}
			if packed.value, err = whispergemm.NewPackedB(state, state); err != nil {
				return errEncoderGEMM
			}
			if packed.out, err = whispergemm.NewPackedB(state, state); err != nil {
				return errEncoderGEMM
			}
			if packed.mlpIn, err = whispergemm.NewPackedB(state, ffn); err != nil {
				return errEncoderGEMM
			}
			if packed.mlpOut, err = whispergemm.NewPackedB(ffn, state); err != nil {
				return errEncoderGEMM
			}
		}
		if packed.query.Pack(block.queryW, state, true) != nil ||
			packed.key.Pack(block.keyW, state, true) != nil ||
			packed.value.Pack(block.valueW, state, true) != nil ||
			packed.out.Pack(block.outW, state, true) != nil ||
			packed.mlpIn.Pack(block.mlpInW, state, true) != nil ||
			packed.mlpOut.Pack(block.mlpOutW, ffn, true) != nil {
			return errEncoderGEMM
		}
	}
	w.packedModel = model
	return nil
}

// tapMajor reorders PyTorch Conv1d weights [out][in][3] into [out][3][in],
// matching the contiguous three-row input windows. It allocates once per
// model binding.
func tapMajor(weights []float32, out, in int) []float32 {
	reordered := make([]float32, len(weights))
	for o := 0; o < out; o++ {
		for c := 0; c < in; c++ {
			for j := 0; j < 3; j++ {
				reordered[o*in*3+j*in+c] = weights[o*in*3+c*3+j]
			}
		}
	}
	return reordered
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
	blocks         []encoderBlockWeights
	finalNormW     []float32
	finalNormB     []float32
}

type encoderBlockNames struct {
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
}

// encoderNames builds tensor names; binding happens once per workspace and
// model, never on a warm inference call.
func encoderNames(layer int) encoderBlockNames {
	p := fmt.Sprintf("encoder.blocks.%d.", layer)
	return encoderBlockNames{
		queryW: p + "attn.query.weight", queryB: p + "attn.query.bias",
		keyW:   p + "attn.key.weight",
		valueW: p + "attn.value.weight", valueB: p + "attn.value.bias",
		outW: p + "attn.out.weight", outB: p + "attn.out.bias",
		attnNormW: p + "attn_ln.weight", attnNormB: p + "attn_ln.bias",
		mlpInW: p + "mlp.0.weight", mlpInB: p + "mlp.0.bias",
		mlpOutW: p + "mlp.2.weight", mlpOutB: p + "mlp.2.bias",
		mlpNormW: p + "mlp_ln.weight", mlpNormB: p + "mlp_ln.bias",
	}
}

func bindEncoderWeights(m *Model) (encoderWeights, bool) {
	d := m.dims
	state, ffn := d.AudioState, 4*d.AudioState
	w := encoderWeights{
		blocks:     make([]encoderBlockWeights, d.AudioLayers),
		conv1W:     m.tensor("encoder.conv1.weight"),
		conv1B:     m.tensor("encoder.conv1.bias"),
		conv2W:     m.tensor("encoder.conv2.weight"),
		conv2B:     m.tensor("encoder.conv2.bias"),
		positions:  m.tensor("encoder.positional_embedding"),
		finalNormW: m.tensor("encoder.ln_post.weight"),
		finalNormB: m.tensor("encoder.ln_post.bias"),
	}
	if len(w.conv1W) != state*MelBins*3 || len(w.conv1B) != state ||
		len(w.conv2W) != state*state*3 || len(w.conv2B) != state ||
		len(w.positions) != AudioFrames*state || len(w.finalNormW) != state || len(w.finalNormB) != state {
		return encoderWeights{}, false
	}
	for i := range w.blocks {
		names := encoderNames(i)
		b := &w.blocks[i]
		b.queryW, b.queryB = m.tensor(names.queryW), m.tensor(names.queryB)
		b.keyW = m.tensor(names.keyW)
		b.valueW, b.valueB = m.tensor(names.valueW), m.tensor(names.valueB)
		b.outW, b.outB = m.tensor(names.outW), m.tensor(names.outB)
		b.attnNormW, b.attnNormB = m.tensor(names.attnNormW), m.tensor(names.attnNormB)
		b.mlpInW, b.mlpInB = m.tensor(names.mlpInW), m.tensor(names.mlpInB)
		b.mlpOutW, b.mlpOutB = m.tensor(names.mlpOutW), m.tensor(names.mlpOutB)
		b.mlpNormW, b.mlpNormB = m.tensor(names.mlpNormW), m.tensor(names.mlpNormB)
		if len(b.queryW) != state*state || len(b.queryB) != state ||
			len(b.keyW) != state*state || len(b.valueW) != state*state || len(b.valueB) != state ||
			len(b.outW) != state*state || len(b.outB) != state ||
			len(b.attnNormW) != state || len(b.attnNormB) != state ||
			len(b.mlpInW) != ffn*state || len(b.mlpInB) != ffn ||
			len(b.mlpOutW) != state*ffn || len(b.mlpOutB) != state ||
			len(b.mlpNormW) != state || len(b.mlpNormB) != state {
			return encoderWeights{}, false
		}
	}
	return w, true
}

func (w *EncoderWorkspace) valid() bool {
	state := w.dims.AudioState
	return state > 0 && len(w.conv1) >= MelFrames*state && len(w.conv1Pad) >= (MelFrames+1)*state &&
		len(w.convColumns) >= (MelFrames+2)*MelBins &&
		len(w.normalized) >= AudioFrames*state &&
		len(w.q) >= AudioFrames*state &&
		len(w.k) >= AudioFrames*state && len(w.v) >= AudioFrames*state &&
		len(w.feedForward) >= AudioFrames*4*state && len(w.packed) == w.dims.AudioLayers &&
		w.conv1Weight != nil && w.conv2Weight != nil && w.attention != nil && w.gemm != nil
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
