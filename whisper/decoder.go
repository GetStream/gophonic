// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"fmt"
	"math"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

var (
	ErrDecoderNilModel       = errors.New("whisper: nil model")
	ErrDecoderNilScratch     = errors.New("whisper: nil decoder scratch")
	ErrDecoderNotStarted     = errors.New("whisper: decoder has not been started")
	ErrDecoderEncoderShape   = errors.New("whisper: encoder features must contain 1 to 1500 time-major frames of 384 values")
	ErrDecoderTokenRange     = errors.New("whisper: token ID is outside the model vocabulary")
	ErrDecoderPosition       = errors.New("whisper: decoder position is out of sequence or exceeds the text context")
	ErrDecoderLogitsTooSmall = errors.New("whisper: logits output is smaller than the vocabulary")
	ErrDecoderPromptEmpty    = errors.New("whisper: greedy decoding needs at least one prompt token")
	ErrDecoderPromptTooLong  = errors.New("whisper: prompt exceeds the text context")
	ErrDecoderOutputTooSmall = errors.New("whisper: output cannot hold the prompt")
)

const layerNormEpsilon = float32(1e-5)

type decoderLinearNames struct {
	weight string
	bias   string
}

type decoderNormNames struct {
	weight string
	bias   string
}

type decoderLayerNames struct {
	selfQ, selfK, selfV, selfOut     decoderLinearNames
	selfNorm                         decoderNormNames
	crossQ, crossK, crossV, crossOut decoderLinearNames
	crossNorm                        decoderNormNames
	mlpIn, mlpOut                    decoderLinearNames
	mlpNorm                          decoderNormNames
}

type decoderTensorNames struct {
	tokenEmbedding, positionEmbedding string
	finalNorm                         decoderNormNames
	layers                            [TextLayers]decoderLayerNames
}

// Names are prepared once at package initialization. Decode calls therefore
// perform no formatting or string construction while resolving model weights.
var decoderNames = makeDecoderTensorNames()

func makeDecoderTensorNames() decoderTensorNames {
	var names decoderTensorNames
	names.tokenEmbedding = "decoder.token_embedding.weight"
	names.positionEmbedding = "decoder.positional_embedding"
	names.finalNorm = decoderNormNames{"decoder.ln.weight", "decoder.ln.bias"}
	for i := range names.layers {
		p := fmt.Sprintf("decoder.blocks.%d.", i)
		linear := func(s string) decoderLinearNames {
			return decoderLinearNames{p + s + ".weight", p + s + ".bias"}
		}
		norm := func(s string) decoderNormNames {
			return decoderNormNames{p + s + ".weight", p + s + ".bias"}
		}
		names.layers[i] = decoderLayerNames{
			selfQ:     linear("attn.query"),
			selfK:     decoderLinearNames{weight: p + "attn.key.weight"},
			selfV:     linear("attn.value"),
			selfOut:   linear("attn.out"),
			selfNorm:  norm("attn_ln"),
			crossQ:    linear("cross_attn.query"),
			crossK:    decoderLinearNames{weight: p + "cross_attn.key.weight"},
			crossV:    linear("cross_attn.value"),
			crossOut:  linear("cross_attn.out"),
			crossNorm: norm("cross_attn_ln"),
			mlpIn:     linear("mlp.0"),
			mlpOut:    linear("mlp.2"),
			mlpNorm:   norm("mlp_ln"),
		}
	}
	return names
}

type decoderLinear struct {
	weight []float32 // [out,in], PyTorch row-major
	bias   []float32 // [out], nil only for attention key projections
}

type decoderNorm struct {
	weight []float32
	bias   []float32
}

type decoderLayerWeights struct {
	selfQ, selfK, selfV, selfOut     decoderLinear
	selfNorm                         decoderNorm
	crossQ, crossK, crossV, crossOut decoderLinear
	crossNorm                        decoderNorm
	mlpIn, mlpOut                    decoderLinear
	mlpNorm                          decoderNorm
}

type decoderWeights struct {
	tokenEmbedding []float32 // [VocabSize,TextState], also the output projection
	position       []float32 // [TextContext,TextState]
	finalNorm      decoderNorm
	layers         [TextLayers]decoderLayerWeights
}

// DecoderScratch owns the incremental self-attention KV cache, projected
// cross-attention KV cache, and all per-token work buffers. It is mutable and
// must not be shared by concurrent decodes. Allocate it once per worker.
//
// Its constructor allocates about 30 MiB for tiny.en, mostly KV caches,
// with 4.5 MiB of packed cross-projection weights and a reusable 2.2 MiB
// value-projection buffer. The vocabulary uses the model's original weights.
type DecoderScratch struct {
	weights     decoderWeights
	weightsFor  *Model
	packedFor   *Model
	crossKey    [TextLayers]*whispergemm.PackedB
	crossValue  [TextLayers]*whispergemm.PackedB
	selfKey     []float32 // [layer,position,state]
	selfValue   []float32 // [layer,state,position], contiguous reduction dimension
	crossKeys   []float32 // [layer,audio-position,state]
	crossValues []float32 // [layer,state,audio-position]
	crossTemp   []float32 // [audio-position,state], reused when projecting values
	x           []float32
	normalized  []float32
	query       []float32
	scaledQuery []float32
	key         []float32
	value       []float32
	context     []float32
	projected   []float32
	mlp         []float32
	scores      []float32
	logits      []float32
	gemm        *whispergemm.Executor // optional borrowed executor; scratch never closes it
	vocabulary  decoderVocabularyProjection
	attention   decoderAttentionOperation
	audioFrames int
	nextPos     int
	ready       bool
}

type decoderVocabularyProjection struct {
	weight, input, output []float32
}

type decoderAttentionOperation struct {
	dst, scaledQuery, keys, values, scores []float32
	frames, valueStride                    int
}

func (op *decoderAttentionOperation) ApplyRows(start, end int) {
	attentionCachedInto(op.dst, op.scaledQuery, op.keys, op.values, op.frames, op.valueStride, op.scores, start, end)
}

func (p *decoderVocabularyProjection) ApplyRows(start, end int) {
	start, end = start*4, min(end*4, VocabSize)
	if err := whispergemm.MulVector(p.output[start:end], p.weight[start*TextState:], TextState, p.input, end-start); err != nil {
		panic(err) // Bound model and fixed decoder shapes have already been checked.
	}
}

// NewDecoderScratch allocates reusable state for one tiny.en decoder worker.
// The first BeginDecode on a model also packs that model's cross-attention
// key/value projections into the scratch; subsequent runs with the same model
// reuse the packed weights.
func NewDecoderScratch() *DecoderScratch {
	s := &DecoderScratch{
		selfKey:     make([]float32, TextLayers*TextContext*TextState),
		selfValue:   make([]float32, TextLayers*TextContext*TextState),
		crossKeys:   make([]float32, TextLayers*AudioFrames*AudioState),
		crossValues: make([]float32, TextLayers*AudioFrames*AudioState),
		crossTemp:   make([]float32, AudioFrames*AudioState),
		x:           make([]float32, TextState),
		normalized:  make([]float32, TextState),
		query:       make([]float32, TextState),
		scaledQuery: make([]float32, TextState),
		key:         make([]float32, TextState),
		value:       make([]float32, TextState),
		context:     make([]float32, TextState),
		projected:   make([]float32, TextState),
		mlp:         make([]float32, 4*TextState),
		scores:      make([]float32, TextHeads*max(AudioFrames, TextContext)),
		logits:      make([]float32, VocabSize),
	}
	for i := 0; i < TextLayers; i++ {
		// These fixed positive dimensions cannot fail validation or overflow.
		s.crossKey[i], _ = whispergemm.NewPackedB(TextState, TextState)
		s.crossValue[i], _ = whispergemm.NewPackedB(TextState, TextState)
	}
	return s
}

// BeginDecode prepares the reusable decoder for one encoded audio sequence.
// It projects and caches cross-attention keys and values once; token steps then
// reuse them. Encoder data is time-major [frames,TextState], with 1..1500
// frames. A normal Whisper tiny.en encoder produces exactly 1500 frames.
func (m *Model) BeginDecode(encoder []float32, s *DecoderScratch) error {
	if m == nil {
		return ErrDecoderNilModel
	}
	if s == nil {
		return ErrDecoderNilScratch
	}
	if len(encoder) == 0 || len(encoder)%AudioState != 0 || len(encoder)/AudioState > AudioFrames {
		return ErrDecoderEncoderShape
	}
	if s.crossKey[0] == nil || s.crossValue[0] == nil {
		return errors.New("whisper: decoder scratch is not initialized")
	}
	s.ready = false
	if s.weightsFor != m {
		if err := loadDecoderWeights(m, &s.weights); err != nil {
			return err
		}
		s.weightsFor = m
	}
	if s.packedFor != m {
		for layer := 0; layer < TextLayers; layer++ {
			w := &s.weights.layers[layer]
			if err := s.crossKey[layer].Pack(w.crossK.weight, TextState, true); err != nil {
				return err
			}
			if err := s.crossValue[layer].Pack(w.crossV.weight, TextState, true); err != nil {
				return err
			}
		}
		s.packedFor = m
	}

	frames := len(encoder) / AudioState
	scale := float32(math.Pow(float64(TextState/TextHeads), -0.25))
	for layer := 0; layer < TextLayers; layer++ {
		base := layer * AudioFrames * AudioState
		keys := s.crossKeys[base : base+frames*AudioState]
		values := s.crossTemp[:frames*AudioState]
		if err := s.multiply(s.crossKey[layer], keys, AudioState, encoder, AudioState, frames); err != nil {
			return err
		}
		if err := s.multiply(s.crossValue[layer], values, AudioState, encoder, AudioState, frames); err != nil {
			return err
		}
		bias := s.weights.layers[layer].crossV.bias
		for d := 0; d < AudioState; d++ {
			valueRow := s.crossValues[base+d*AudioFrames : base+d*AudioFrames+frames]
			for frame := range valueRow {
				valueRow[frame] = values[frame*AudioState+d] + bias[d]
			}
		}
		for i := range keys {
			keys[i] *= scale
		}
	}
	s.audioFrames = frames
	s.nextPos = 0 // Old self-KV entries at positions >= 0 are overwritten before read.
	s.ready = true
	return nil
}

func (s *DecoderScratch) multiply(b *whispergemm.PackedB, dst []float32, dstStride int, a []float32, aStride, rows int) error {
	if s.gemm != nil {
		return s.gemm.Mul(b, dst, dstStride, a, aStride, rows)
	}
	return b.Mul(dst, dstStride, a, aStride, rows)
}

// LogitsForTokenInto runs one causal decoder position and writes the next-token
// logits into caller-owned storage. Calls must follow BeginDecode and use
// positions 0,1,... without gaps. The logits slice must hold VocabSize values.
// This operation is allocation-free after NewDecoderScratch and BeginDecode.
func (m *Model) LogitsForTokenInto(tokenID, position int, s *DecoderScratch, logits []float32) error {
	if m == nil {
		return ErrDecoderNilModel
	}
	if s == nil {
		return ErrDecoderNilScratch
	}
	if !s.ready || s.weightsFor != m {
		return ErrDecoderNotStarted
	}
	if tokenID < 0 || tokenID >= VocabSize {
		return ErrDecoderTokenRange
	}
	if position != s.nextPos || position < 0 || position >= TextContext {
		return ErrDecoderPosition
	}
	if len(logits) < VocabSize {
		return ErrDecoderLogitsTooSmall
	}

	state := TextState
	weights := &s.weights
	embedStart := tokenID * state
	posStart := position * state
	for d := 0; d < state; d++ {
		s.x[d] = weights.tokenEmbedding[embedStart+d] + weights.position[posStart+d]
	}

	for layer := 0; layer < TextLayers; layer++ {
		lw := &weights.layers[layer]

		// Residual self attention: x += attn(attn_ln(x)). Projected K/V are
		// stored at this position before causal attention reads the cache.
		layerNorm(s.normalized, s.x, lw.selfNorm.weight, lw.selfNorm.bias)
		linearInto(s.query, s.normalized, lw.selfQ.weight, lw.selfQ.bias, state, state)
		linearInto(s.key, s.normalized, lw.selfK.weight, nil, state, state)
		linearInto(s.value, s.normalized, lw.selfV.weight, lw.selfV.bias, state, state)
		cacheBase := (layer*TextContext + position) * state
		scale := float32(math.Pow(float64(state/TextHeads), -0.25))
		for d := 0; d < state; d++ {
			s.selfKey[cacheBase+d] = s.key[d] * scale
			s.selfValue[(layer*state+d)*TextContext+position] = s.value[d]
		}
		layerBase := layer * TextContext * state
		cachedKeys := s.selfKey[layerBase : cacheBase+state]
		cachedValues := s.selfValue[layerBase : layerBase+TextContext*state]
		if err := s.attend(cachedKeys, cachedValues, position+1, TextContext); err != nil {
			return err
		}
		linearInto(s.projected, s.context, lw.selfOut.weight, lw.selfOut.bias, state, state)
		addInto(s.x, s.projected)

		// Residual cross attention: x += cross_attn(cross_attn_ln(x), audio).
		layerNorm(s.normalized, s.x, lw.crossNorm.weight, lw.crossNorm.bias)
		linearInto(s.query, s.normalized, lw.crossQ.weight, lw.crossQ.bias, state, state)
		crossBase := layer * AudioFrames * AudioState
		crossEnd := crossBase + s.audioFrames*AudioState
		if err := s.attend(s.crossKeys[crossBase:crossEnd], s.crossValues[crossBase:crossBase+AudioFrames*state], s.audioFrames, AudioFrames); err != nil {
			return err
		}
		linearInto(s.projected, s.context, lw.crossOut.weight, lw.crossOut.bias, state, state)
		addInto(s.x, s.projected)

		// Residual MLP: x += mlp(mlp_ln(x)); PyTorch nn.GELU uses the exact erf form.
		layerNorm(s.normalized, s.x, lw.mlpNorm.weight, lw.mlpNorm.bias)
		linearInto(s.mlp, s.normalized, lw.mlpIn.weight, lw.mlpIn.bias, state, 4*state)
		geluExactInto(s.mlp)
		linearInto(s.projected, s.mlp, lw.mlpOut.weight, lw.mlpOut.bias, 4*state, state)
		addInto(s.x, s.projected)
	}

	layerNorm(s.normalized, s.x, weights.finalNorm.weight, weights.finalNorm.bias)
	if s.gemm == nil {
		if err := whispergemm.MulVector(logits, weights.tokenEmbedding, state, s.normalized, VocabSize); err != nil {
			return err
		}
	} else {
		s.vocabulary = decoderVocabularyProjection{weight: weights.tokenEmbedding, input: s.normalized, output: logits}
		err := s.gemm.Rows(&s.vocabulary, (VocabSize+3)/4, 256)
		s.vocabulary = decoderVocabularyProjection{}
		if err != nil {
			return err
		}
	}
	s.nextPos++
	return nil
}

// GreedyDecodeInto runs raw greedy argmax decoding. prompt is the caller's
// initial token sequence; output receives prompt followed by generated tokens,
// including eosID when it is selected. It returns the number of valid output
// IDs, stopping at EOS or when output is full. Use GreedyPolicy in the outer
// generation loop when Whisper suppression or task-specific filtering is
// required; this convenience method intentionally performs unfiltered argmax.
//
// Once scratch has been created, successful calls allocate no memory.
func (m *Model) GreedyDecodeInto(encoder []float32, prompt, output []int, s *DecoderScratch, eosID int) (int, error) {
	if len(prompt) == 0 {
		return 0, ErrDecoderPromptEmpty
	}
	if len(prompt) > TextContext {
		return 0, ErrDecoderPromptTooLong
	}
	if len(output) < len(prompt) {
		return 0, ErrDecoderOutputTooSmall
	}
	if eosID < 0 || eosID >= VocabSize {
		return 0, ErrDecoderTokenRange
	}
	for _, id := range prompt {
		if id < 0 || id >= VocabSize {
			return 0, ErrDecoderTokenRange
		}
	}
	if err := m.BeginDecode(encoder, s); err != nil {
		return 0, err
	}
	copy(output, prompt)
	n := len(prompt)
	logits := s.logits
	for position := 0; position < len(prompt); position++ {
		if err := m.LogitsForTokenInto(output[position], position, s, logits); err != nil {
			return n, err
		}
	}
	for n < len(output) {
		next := argmax(logits)
		output[n] = next
		n++
		if next == eosID || s.nextPos >= TextContext {
			return n, nil
		}
		if err := m.LogitsForTokenInto(next, s.nextPos, s, logits); err != nil {
			return n, err
		}
	}
	return n, nil
}

func loadDecoderWeights(m *Model, dst *decoderWeights) error {
	var err error
	if dst.tokenEmbedding, err = checkedTensor(m, decoderNames.tokenEmbedding, VocabSize*TextState); err != nil {
		return err
	}
	if dst.position, err = checkedTensor(m, decoderNames.positionEmbedding, TextContext*TextState); err != nil {
		return err
	}
	if err = loadNorm(m, decoderNames.finalNorm, &dst.finalNorm); err != nil {
		return err
	}
	for layer := 0; layer < TextLayers; layer++ {
		n := decoderNames.layers[layer]
		w := &dst.layers[layer]
		if err = loadLinear(m, n.selfQ, TextState, TextState, &w.selfQ); err != nil {
			return err
		}
		if err = loadLinear(m, n.selfK, TextState, TextState, &w.selfK); err != nil {
			return err
		}
		if err = loadLinear(m, n.selfV, TextState, TextState, &w.selfV); err != nil {
			return err
		}
		if err = loadLinear(m, n.selfOut, TextState, TextState, &w.selfOut); err != nil {
			return err
		}
		if err = loadNorm(m, n.selfNorm, &w.selfNorm); err != nil {
			return err
		}
		if err = loadLinear(m, n.crossQ, TextState, TextState, &w.crossQ); err != nil {
			return err
		}
		if err = loadLinear(m, n.crossK, TextState, TextState, &w.crossK); err != nil {
			return err
		}
		if err = loadLinear(m, n.crossV, TextState, TextState, &w.crossV); err != nil {
			return err
		}
		if err = loadLinear(m, n.crossOut, TextState, TextState, &w.crossOut); err != nil {
			return err
		}
		if err = loadNorm(m, n.crossNorm, &w.crossNorm); err != nil {
			return err
		}
		if err = loadLinear(m, n.mlpIn, TextState, 4*TextState, &w.mlpIn); err != nil {
			return err
		}
		if err = loadLinear(m, n.mlpOut, 4*TextState, TextState, &w.mlpOut); err != nil {
			return err
		}
		if err = loadNorm(m, n.mlpNorm, &w.mlpNorm); err != nil {
			return err
		}
	}
	return nil
}

func loadLinear(m *Model, names decoderLinearNames, in, out int, dst *decoderLinear) error {
	var err error
	if dst.weight, err = checkedTensor(m, names.weight, in*out); err != nil {
		return err
	}
	if names.bias == "" {
		dst.bias = nil
		return nil
	}
	if dst.bias, err = checkedTensor(m, names.bias, out); err != nil {
		return err
	}
	return nil
}

func loadNorm(m *Model, names decoderNormNames, dst *decoderNorm) error {
	var err error
	if dst.weight, err = checkedTensor(m, names.weight, TextState); err != nil {
		return err
	}
	if dst.bias, err = checkedTensor(m, names.bias, TextState); err != nil {
		return err
	}
	return nil
}

func checkedTensor(m *Model, name string, size int) ([]float32, error) {
	t := m.tensor(name)
	if len(t) != size {
		return nil, fmt.Errorf("whisper: tensor %q has %d values; want %d", name, len(t), size)
	}
	return t, nil
}

func linearInto(dst, x, weight, bias []float32, in, out int) {
	if err := whispergemm.MulVector(dst, weight, in, x[:in], out); err != nil {
		panic(err) // Internal callers use validated, fixed model dimensions.
	}
	if bias != nil {
		for o := 0; o < out; o++ {
			dst[o] += bias[o]
		}
	}
}

func layerNorm(dst, x, weight, bias []float32) {
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	var variance float32
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float32(len(x))
	invStd := float32(1 / math.Sqrt(float64(variance+layerNormEpsilon)))
	for i, v := range x {
		dst[i] = ((v-mean)*invStd)*weight[i] + bias[i]
	}
}

func attentionInto(dst, query, keys, values []float32, frames, heads int, scores []float32) {
	state := len(query)
	headSize := state / heads
	scale := float32(math.Pow(float64(headSize), -0.25))
	for head := 0; head < heads; head++ {
		start := head * headSize
		end := start + headSize
		maxScore := float32(math.Inf(-1))
		for frame := 0; frame < frames; frame++ {
			base := frame * state
			var score float32
			for d := start; d < end; d++ {
				q := query[d] * scale
				k := keys[base+d] * scale
				score += q * k
			}
			scores[frame] = score
			if score > maxScore {
				maxScore = score
			}
		}
		var sum float32
		for frame := 0; frame < frames; frame++ {
			p := float32(math.Exp(float64(scores[frame] - maxScore)))
			scores[frame] = p
			sum += p
		}
		invSum := float32(1 / sum)
		for d := start; d < end; d++ {
			var value float32
			for frame := 0; frame < frames; frame++ {
				p := scores[frame] * invSum
				value += p * values[frame*state+d]
			}
			dst[d] = value
		}
	}
}

// Cached keys already contain Whisper's head-size^-1/4 scale. Values keep
// their reduction axis contiguous so the same GEMV kernel handles both
// attention products without repacking at each generated token.
func (s *DecoderScratch) attend(keys, values []float32, frames, valueStride int) error {
	const headSize = TextState / TextHeads
	scale := float32(math.Pow(float64(headSize), -0.25))
	for d, value := range s.query {
		s.scaledQuery[d] = value * scale
	}
	s.attention = decoderAttentionOperation{dst: s.context, scaledQuery: s.scaledQuery, keys: keys, values: values, scores: s.scores, frames: frames, valueStride: valueStride}
	var err error
	if s.gemm != nil && frames >= 128 {
		err = s.gemm.Rows(&s.attention, TextHeads, 1)
	} else {
		s.attention.ApplyRows(0, TextHeads)
	}
	s.attention = decoderAttentionOperation{}
	return err
}

func attentionCachedInto(dst, scaledQuery, keys, values []float32, frames, valueStride int, scores []float32, firstHead, lastHead int) {
	const headSize = TextState / TextHeads
	for head := firstHead; head < lastHead; head++ {
		probabilities := scores[head*AudioFrames : head*AudioFrames+frames]
		start, end := head*headSize, (head+1)*headSize
		if err := whispergemm.MulVector(probabilities, keys[start:], TextState, scaledQuery[start:end], frames); err != nil {
			panic(err)
		}
		maxScore := float32(math.Inf(-1))
		for _, score := range probabilities {
			if score > maxScore {
				maxScore = score
			}
		}
		var sum float32
		for frame, score := range probabilities {
			p := float32(math.Exp(float64(score - maxScore)))
			probabilities[frame] = p
			sum += p
		}
		invSum := float32(1 / sum)
		for frame := range probabilities {
			probabilities[frame] *= invSum
		}
		if err := whispergemm.MulVector(dst[start:end], values[start*valueStride:], valueStride, probabilities, headSize); err != nil {
			panic(err)
		}
	}
}

func geluExactInto(x []float32) {
	const invSqrt2 = 0.7071067811865475244
	for i, v := range x {
		erf := float32(math.Erf(float64(v) * invSqrt2))
		x[i] = (float32(0.5) * v) * (float32(1) + erf)
	}
}

func addInto(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func argmax(logits []float32) int {
	best := 0
	bestValue := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestValue {
			best, bestValue = i, logits[i]
		}
	}
	return best
}
