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

var (
	ErrDecoderNilModel       = errors.New("whisper: nil model")
	errDecoderNilScratch     = errors.New("whisper: nil decoder scratch")
	errDecoderNotStarted     = errors.New("whisper: decoder has not been started")
	errDecoderEncoderShape   = errors.New("whisper: encoder features must contain 1 to 1500 time-major frames of 384 values")
	errDecoderTokenRange     = errors.New("whisper: token ID is outside the model vocabulary")
	errDecoderPosition       = errors.New("whisper: decoder position is out of sequence or exceeds the text context")
	errDecoderLogitsTooSmall = errors.New("whisper: logits output is smaller than the vocabulary")
	errDecoderPromptEmpty    = errors.New("whisper: greedy decoding needs at least one prompt token")
	errDecoderPromptTooLong  = errors.New("whisper: prompt exceeds the text context")
	errDecoderOutputTooSmall = errors.New("whisper: output cannot hold the prompt")
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
	layers                            []decoderLayerNames
}

// maxTextLayers bounds the decoder depth of supported checkpoints.
const maxTextLayers = 64

// Names are prepared once at package initialization. Decode calls therefore
// perform no formatting or string construction while resolving model weights.
var decoderNames = makeDecoderTensorNames()

func makeDecoderTensorNames() decoderTensorNames {
	var names decoderTensorNames
	names.layers = make([]decoderLayerNames, maxTextLayers)
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
	weight []float32                 // [out,in], PyTorch row-major
	bias   []float32                 // [out], nil only for attention key projections
	packed *whispergemm.PackedVector // shared model packing; nil without SME
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
	tokenEmbedding []float32                 // [VocabSize,TextState], also the output projection
	vocabulary     *whispergemm.PackedVector // shared packing of tokenEmbedding; nil without SME
	position       []float32                 // [TextContext,TextState]
	finalNorm      decoderNorm
	layers         []decoderLayerWeights
}

// decoderScratch owns the incremental self-attention KV cache, projected
// cross-attention KV cache, and all per-token work buffers. It is mutable and
// must not be shared by concurrent decodes. Allocate it once per worker.
//
// Immutable weights are shared by all lanes on the model. With packed
// cross-attention caches, raw projection scratch holds just one layer and is
// reused while preparing the next layer; it never duplicates the full cache.
type decoderScratch struct {
	weights    decoderWeights
	weightsFor *Model
	packedFor  *Model
	dims       modelDims
	// crossKeyVec and crossValueVec hold each layer and head's cross-attention
	// cache packed for the streaming GEMV kernel; nil without SME.
	crossKeyVec   []*whispergemm.PackedVector
	crossValueVec []*whispergemm.PackedVector
	encoderT      *whispergemm.PackedB        // encoder^T for transposed key projection
	layerKeys     []*whispergemm.PackedVector // current layer's packed heads, if any
	layerValues   []*whispergemm.PackedVector
	crossKey      []*whispergemm.PackedB
	crossValue    []*whispergemm.PackedB
	selfKey       []float32 // [layer,position,state]
	selfValue     []float32 // [layer,state,position], contiguous reduction dimension
	crossKeys     []float32 // [layer,audio-position,state]
	crossValues   []float32 // [layer,state,audio-position]
	crossTemp     []float32 // [audio-position,state], reused when projecting values
	x             []float32
	normalized    []float32
	query         []float32
	scaledQuery   []float32
	key           []float32
	value         []float32
	context       []float32
	projected     []float32
	mlp           []float32
	scores        []float32
	logits        []float32
	gemm          *whispergemm.Executor // optional borrowed executor; scratch never closes it
	vocabulary    decoderVocabularyProjection
	attention     decoderAttentionOperation
	audioFrames   int
	alignment     *alignmentCapture
	nextPos       int
	ready         bool
}

type decoderVocabularyProjection struct {
	weight, input, output []float32
	packed                *whispergemm.PackedVector
	shards, state         int
}

// vocabularyShards bounds concurrent streaming-matrix work on the vocabulary.
var vocabularyShards = 8

type decoderAttentionOperation struct {
	dst, scaledQuery, keys, values, scores []float32
	frames, valueStride, heads             int
	keyVec, valueVec                       []*whispergemm.PackedVector
}

func (op *decoderAttentionOperation) ApplyRows(start, end int) {
	if op.keyVec != nil {
		attentionPackedInto(op.dst, op.scaledQuery, op.keyVec, op.valueVec, op.frames, op.heads, op.scores, start, end)
		return
	}
	attentionCachedInto(op.dst, op.scaledQuery, op.keys, op.values, op.frames, op.valueStride, op.heads, op.scores, start, end)
}

func (p *decoderVocabularyProjection) ApplyRows(start, end int) {
	if p.packed != nil {
		chunks := p.packed.Chunks()
		for shard := start; shard < end; shard++ {
			first, last := shard*chunks/p.shards, (shard+1)*chunks/p.shards
			if err := p.packed.MulChunks(p.output, p.input, first, last); err != nil {
				panic(err) // Bound model and fixed decoder shapes have already been checked.
			}
		}
		return
	}
	start, end = start*4, min(end*4, vocabSize)
	if err := whispergemm.MulVector(p.output[start:end], p.weight[start*p.state:], p.state, p.input, end-start); err != nil {
		panic(err) // Bound model and fixed decoder shapes have already been checked.
	}
}

// newTinyDecoderScratch allocates reusable state for one tiny.en decoder worker.
// The first beginDecode on a model also packs that model's cross-attention
// key/value projections into the scratch; subsequent runs with the same model
// reuse the packed weights.
func newTinyDecoderScratch() *decoderScratch {
	return newDecoderScratch(tinyENDims)
}

func newDecoderScratch(d modelDims) *decoderScratch {
	textState, textLayers, textHeads, audioState := d.TextState, d.TextLayers, d.TextHeads, d.AudioState
	crossLayers := textLayers
	if whispergemm.PackedVectorAccelerated() {
		crossLayers = 1
	}
	s := &decoderScratch{
		dims:        d,
		crossKey:    make([]*whispergemm.PackedB, textLayers),
		crossValue:  make([]*whispergemm.PackedB, textLayers),
		selfKey:     make([]float32, textLayers*textContext*textState),
		selfValue:   make([]float32, textLayers*textContext*textState),
		crossKeys:   make([]float32, crossLayers*audioFrames*audioState),
		crossValues: make([]float32, crossLayers*audioFrames*audioState),
		crossTemp:   make([]float32, audioFrames*audioState),
		x:           make([]float32, textState),
		normalized:  make([]float32, textState),
		query:       make([]float32, textState),
		scaledQuery: make([]float32, textState),
		key:         make([]float32, textState),
		value:       make([]float32, textState),
		context:     make([]float32, textState),
		projected:   make([]float32, textState),
		mlp:         make([]float32, 4*textState),
		scores:      make([]float32, textHeads*max(audioFrames, textContext)),
		logits:      make([]float32, vocabSize),
	}
	for i := 0; i < textLayers; i++ {
		// These fixed positive dimensions cannot fail validation or overflow.
		s.crossKey[i], _ = whispergemm.NewPackedB(textState, textState)
		s.crossValue[i], _ = whispergemm.NewPackedB(textState, textState)
	}
	if whispergemm.PackedVectorAccelerated() {
		headSize := textState / textHeads
		n := textLayers * textHeads
		s.crossKeyVec = make([]*whispergemm.PackedVector, n)
		s.crossValueVec = make([]*whispergemm.PackedVector, n)
		for i := range n {
			// One allocation fits either [frames,head] or [head,frames].
			s.crossKeyVec[i], _ = whispergemm.NewPackedVectorFP32(audioFrames, headSize)
			s.crossValueVec[i], _ = whispergemm.NewPackedVectorFP32(audioFrames, headSize)
		}
	}
	return s
}

// beginDecode prepares the reusable decoder for one encoded audio sequence.
// It projects and caches cross-attention keys and values once; token steps then
// reuse them. Encoder data is time-major [frames,TextState], with 1..1500
// frames. A normal Whisper tiny.en encoder produces exactly 1500 frames.
func (m *Model) beginDecode(encoder []float32, s *decoderScratch) error {
	defer runtime.KeepAlive(s)
	defer runtime.KeepAlive(m)
	if m == nil {
		return ErrDecoderNilModel
	}
	if s == nil {
		return errDecoderNilScratch
	}
	d := s.dims
	if m.dims != d {
		return fmt.Errorf("whisper: decoder scratch dimensions %+v differ from model %+v", d, m.dims)
	}
	textState, textLayers, textHeads, audioState := d.TextState, d.TextLayers, d.TextHeads, d.AudioState
	if len(encoder) == 0 || len(encoder)%audioState != 0 || len(encoder)/audioState > audioFrames {
		return errDecoderEncoderShape
	}
	if len(s.crossKey) == 0 || s.crossKey[0] == nil || s.crossValue[0] == nil {
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
		for layer := 0; layer < textLayers; layer++ {
			w := &s.weights.layers[layer]
			if err := s.crossKey[layer].Pack(w.crossK.weight, textState, true); err != nil {
				return err
			}
			if err := s.crossValue[layer].Pack(w.crossV.weight, textState, true); err != nil {
				return err
			}
		}
		s.packedFor = m
	}

	frames := len(encoder) / audioState
	scale := float32(math.Pow(float64(textState/textHeads), -0.25))
	if s.crossKeyVec != nil && frames == audioFrames {
		if err := s.beginPacked(encoder, scale); err != nil {
			return err
		}
		s.audioFrames = frames
		s.nextPos = 0
		s.ready = true
		return nil
	}
	for layer := 0; layer < textLayers; layer++ {
		base := layer * audioFrames * audioState
		if s.crossKeyVec != nil {
			base = 0
		}
		keys := s.crossKeys[base : base+frames*audioState]
		values := s.crossTemp[:frames*audioState]
		if err := s.multiply(s.crossKey[layer], keys, audioState, encoder, audioState, frames); err != nil {
			return err
		}
		if err := s.multiply(s.crossValue[layer], values, audioState, encoder, audioState, frames); err != nil {
			return err
		}
		bias := s.weights.layers[layer].crossV.bias
		for d := 0; d < audioState; d++ {
			valueRow := s.crossValues[base+d*audioFrames : base+d*audioFrames+frames]
			for frame := range valueRow {
				valueRow[frame] = values[frame*audioState+d] + bias[d]
			}
		}
		for i := range keys {
			keys[i] *= scale
		}
		if s.crossKeyVec != nil {
			headSize := textState / textHeads
			for head := 0; head < textHeads; head++ {
				i := layer*textHeads + head
				if err := s.crossKeyVec[i].RepackShape(keys[head*headSize:], audioState, frames, headSize); err != nil {
					return err
				}
				values := s.crossValues[base+head*headSize*audioFrames:]
				if err := s.crossValueVec[i].RepackShape(values, audioFrames, headSize, frames); err != nil {
					return err
				}
			}
		}
	}
	s.audioFrames = frames
	s.nextPos = 0 // Old self-KV entries at positions >= 0 are overwritten before read.
	s.ready = true
	return nil
}

// beginPacked projects the cross-attention cache straight into the per-head
// streaming GEMV layout. Keys are computed transposed, as Wk * encoder^T, so
// each head's frames are contiguous; values keep [frame,state] so each head's
// dimensions are contiguous. The SME GEMM sums every output in the same K
// order either way, so keys are bit-identical to the untransposed product.
func (s *decoderScratch) beginPacked(encoder []float32, scale float32) error {
	d := s.dims
	state, heads := d.TextState, d.TextHeads
	headSize := state / heads
	if s.encoderT == nil {
		var err error
		if s.encoderT, err = whispergemm.NewPackedB(state, audioFrames); err != nil {
			return err
		}
	}
	if err := s.encoderT.Pack(encoder, state, true); err != nil {
		return err
	}
	keysT := s.crossKeys[:state*audioFrames] // scratch: [state, frames]
	values := s.crossTemp[:audioFrames*state]
	for layer := 0; layer < d.TextLayers; layer++ {
		w := &s.weights.layers[layer]
		if err := s.multiply(s.encoderT, keysT, audioFrames, w.crossK.weight, state, state); err != nil {
			return err
		}
		if err := s.multiply(s.crossValue[layer], values, state, encoder, state, audioFrames); err != nil {
			return err
		}
		for head := 0; head < heads; head++ {
			i := layer*heads + head
			if err := s.crossKeyVec[i].RepackColumns(keysT[head*headSize*audioFrames:], audioFrames, audioFrames, headSize, scale, nil); err != nil {
				return err
			}
			if err := s.crossValueVec[i].RepackColumns(values[head*headSize:], state, headSize, audioFrames, 1, w.crossV.bias[head*headSize:]); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *decoderScratch) multiply(b *whispergemm.PackedB, dst []float32, dstStride int, a []float32, aStride, rows int) error {
	if s.gemm != nil {
		return s.gemm.Mul(b, dst, dstStride, a, aStride, rows)
	}
	return b.Mul(dst, dstStride, a, aStride, rows)
}

// logitsForTokenInto runs one causal decoder position and writes the next-token
// logits into caller-owned storage. Calls must follow beginDecode and use
// positions 0,1,... without gaps. The logits slice must hold VocabSize values.
// This operation is allocation-free after newTinyDecoderScratch and beginDecode.
func (m *Model) logitsForTokenInto(tokenID, position int, s *decoderScratch, logits []float32) error {
	return m.decodeTokenInto(tokenID, position, s, logits, true)
}

// decodeTokenInto may omit the vocabulary projection when only the KV state
// or alignment is consumed. This never changes the next position's state.
func (m *Model) decodeTokenInto(tokenID, position int, s *decoderScratch, logits []float32, project bool) error {
	defer runtime.KeepAlive(s)
	if m == nil {
		return ErrDecoderNilModel
	}
	if s == nil {
		return errDecoderNilScratch
	}
	if !s.ready || s.weightsFor != m {
		return errDecoderNotStarted
	}
	if tokenID < 0 || tokenID >= vocabSize {
		return errDecoderTokenRange
	}
	if position != s.nextPos || position < 0 || position >= textContext {
		return errDecoderPosition
	}
	if project && len(logits) < vocabSize {
		return errDecoderLogitsTooSmall
	}

	d := s.dims
	state, textLayers, textHeads, audioState := d.TextState, d.TextLayers, d.TextHeads, d.AudioState
	weights := &s.weights
	embedStart := tokenID * state
	posStart := position * state
	for d := 0; d < state; d++ {
		s.x[d] = weights.tokenEmbedding[embedStart+d] + weights.position[posStart+d]
	}

	var pending []float32 // MLP output awaiting a fused add+norm.
	for layer := 0; layer < textLayers; layer++ {
		lw := &weights.layers[layer]

		// Residual self attention: x += attn(attn_ln(x)). Projected K/V are
		// stored at this position before causal attention reads the cache.
		if pending == nil {
			layerNormInto(s.normalized, s.x, lw.selfNorm.weight, lw.selfNorm.bias)
		} else {
			// Fuse the previous MLP block's residual add with this norm.
			residualNormInto(s.x, s.normalized, pending, lw.selfNorm.weight, lw.selfNorm.bias)
		}
		linearInto(s.query, s.normalized, &lw.selfQ, state, state)
		linearInto(s.key, s.normalized, &lw.selfK, state, state)
		linearInto(s.value, s.normalized, &lw.selfV, state, state)
		cacheBase := (layer*textContext + position) * state
		scale := float32(math.Pow(float64(state/textHeads), -0.25))
		for d := 0; d < state; d++ {
			s.selfKey[cacheBase+d] = s.key[d] * scale
			s.selfValue[(layer*state+d)*textContext+position] = s.value[d]
		}
		layerBase := layer * textContext * state
		cachedKeys := s.selfKey[layerBase : cacheBase+state]
		cachedValues := s.selfValue[layerBase : layerBase+textContext*state]
		if err := s.attend(cachedKeys, cachedValues, position+1, textContext); err != nil {
			return err
		}
		linearInto(s.projected, s.context, &lw.selfOut, state, state)

		// Residual cross attention: x += cross_attn(cross_attn_ln(x), audio).
		// The self-attention residual add is fused with this norm.
		residualNormInto(s.x, s.normalized, s.projected, lw.crossNorm.weight, lw.crossNorm.bias)
		linearInto(s.query, s.normalized, &lw.crossQ, state, state)
		crossBase := layer * audioFrames * audioState
		if s.crossKeyVec != nil {
			crossBase = 0
		}
		crossEnd := crossBase + s.audioFrames*audioState
		s.layerKeys, s.layerValues = nil, nil
		if s.crossKeyVec != nil {
			s.layerKeys = s.crossKeyVec[layer*textHeads : (layer+1)*textHeads]
			s.layerValues = s.crossValueVec[layer*textHeads : (layer+1)*textHeads]
		}
		err := s.attend(s.crossKeys[crossBase:crossEnd], s.crossValues[crossBase:crossBase+audioFrames*audioState], s.audioFrames, audioFrames)
		if err == nil && s.alignment != nil {
			s.alignment.capture(layer, position, s.scores)
		}
		s.layerKeys, s.layerValues = nil, nil
		if err != nil {
			return err
		}
		linearInto(s.projected, s.context, &lw.crossOut, state, state)

		// Residual MLP: x += mlp(mlp_ln(x)); PyTorch nn.GELU uses the exact erf form.
		// The cross-attention residual add is fused with this norm.
		residualNormInto(s.x, s.normalized, s.projected, lw.mlpNorm.weight, lw.mlpNorm.bias)
		linearInto(s.mlp, s.normalized, &lw.mlpIn, state, 4*state)
		geluExactInto(s.mlp)
		linearInto(s.projected, s.mlp, &lw.mlpOut, 4*state, state)
		pending = s.projected
	}

	if !project {
		s.nextPos++
		return nil
	}
	// Fuse the last MLP block's residual add with the final norm.
	residualNormInto(s.x, s.normalized, pending, weights.finalNorm.weight, weights.finalNorm.bias)
	if weights.vocabulary != nil {
		if s.gemm == nil || s.gemm.Workers() < 2 {
			if err := weights.vocabulary.Mul(logits, s.normalized); err != nil {
				return err
			}
		} else {
			// Shards spread this bandwidth-bound product over the
			// performance clusters' matrix units.
			shards := min(s.gemm.Workers(), vocabularyShards)
			s.vocabulary = decoderVocabularyProjection{packed: weights.vocabulary, input: s.normalized, output: logits, shards: shards}
			err := s.gemm.Rows(&s.vocabulary, shards, 1)
			s.vocabulary = decoderVocabularyProjection{}
			if err != nil {
				return err
			}
		}
	} else if s.gemm == nil {
		if err := whispergemm.MulVector(logits, weights.tokenEmbedding, state, s.normalized, vocabSize); err != nil {
			return err
		}
	} else {
		s.vocabulary = decoderVocabularyProjection{state: state, weight: weights.tokenEmbedding, input: s.normalized, output: logits}
		err := s.gemm.Rows(&s.vocabulary, (vocabSize+3)/4, 256)
		s.vocabulary = decoderVocabularyProjection{}
		if err != nil {
			return err
		}
	}
	s.nextPos++
	return nil
}

// greedyDecodeInto runs raw greedy argmax decoding. prompt is the caller's
// initial token sequence; output receives prompt followed by generated tokens,
// including eosID when it is selected. It returns the number of valid output
// IDs, stopping at EOS or when output is full. Use greedyPolicy in the outer
// generation loop when Whisper suppression or task-specific filtering is
// required; this convenience method intentionally performs unfiltered argmax.
//
// Once scratch has been created, successful calls allocate no memory.
func (m *Model) greedyDecodeInto(encoder []float32, prompt, output []int, s *decoderScratch, eosID int) (int, error) {
	if len(prompt) == 0 {
		return 0, errDecoderPromptEmpty
	}
	if len(prompt) > textContext {
		return 0, errDecoderPromptTooLong
	}
	if len(output) < len(prompt) {
		return 0, errDecoderOutputTooSmall
	}
	if eosID < 0 || eosID >= vocabSize {
		return 0, errDecoderTokenRange
	}
	for _, id := range prompt {
		if id < 0 || id >= vocabSize {
			return 0, errDecoderTokenRange
		}
	}
	if err := m.beginDecode(encoder, s); err != nil {
		return 0, err
	}
	copy(output, prompt)
	n := len(prompt)
	logits := s.logits
	for position := 0; position < len(prompt); position++ {
		if err := m.decodeTokenInto(output[position], position, s, logits, position == len(prompt)-1); err != nil {
			return n, err
		}
	}
	for n < len(output) {
		next := argmax(logits)
		output[n] = next
		n++
		if next == eosID || s.nextPos >= textContext {
			return n, nil
		}
		if err := m.logitsForTokenInto(next, s.nextPos, s, logits); err != nil {
			return n, err
		}
	}
	return n, nil
}

func loadDecoderWeights(m *Model, dst *decoderWeights) error {
	var err error
	textState, textLayers := m.dims.TextState, m.dims.TextLayers
	if textLayers > maxTextLayers {
		return fmt.Errorf("whisper: %d decoder layers exceed the supported %d", textLayers, maxTextLayers)
	}
	if len(dst.layers) != textLayers {
		dst.layers = make([]decoderLayerWeights, textLayers)
	}
	if dst.tokenEmbedding, err = checkedTensor(m, decoderNames.tokenEmbedding, vocabSize*textState); err != nil {
		return err
	}
	if dst.vocabulary, err = m.packedVector(decoderNames.tokenEmbedding, vocabSize, textState); err != nil {
		return err
	}
	if dst.position, err = checkedTensor(m, decoderNames.positionEmbedding, textContext*textState); err != nil {
		return err
	}
	if err = loadNorm(m, decoderNames.finalNorm, textState, &dst.finalNorm); err != nil {
		return err
	}
	for layer := 0; layer < textLayers; layer++ {
		n := decoderNames.layers[layer]
		w := &dst.layers[layer]
		if err = loadLinear(m, n.selfQ, textState, textState, &w.selfQ); err != nil {
			return err
		}
		if err = loadLinear(m, n.selfK, textState, textState, &w.selfK); err != nil {
			return err
		}
		if err = loadLinear(m, n.selfV, textState, textState, &w.selfV); err != nil {
			return err
		}
		if err = loadLinear(m, n.selfOut, textState, textState, &w.selfOut); err != nil {
			return err
		}
		if err = loadNorm(m, n.selfNorm, textState, &w.selfNorm); err != nil {
			return err
		}
		if err = loadLinear(m, n.crossQ, textState, textState, &w.crossQ); err != nil {
			return err
		}
		if err = loadLinear(m, n.crossK, textState, textState, &w.crossK); err != nil {
			return err
		}
		if err = loadLinear(m, n.crossV, textState, textState, &w.crossV); err != nil {
			return err
		}
		if err = loadLinear(m, n.crossOut, textState, textState, &w.crossOut); err != nil {
			return err
		}
		if err = loadNorm(m, n.crossNorm, textState, &w.crossNorm); err != nil {
			return err
		}
		if err = loadLinear(m, n.mlpIn, textState, 4*textState, &w.mlpIn); err != nil {
			return err
		}
		if err = loadLinear(m, n.mlpOut, 4*textState, textState, &w.mlpOut); err != nil {
			return err
		}
		if err = loadNorm(m, n.mlpNorm, textState, &w.mlpNorm); err != nil {
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
	if dst.packed, err = m.packedVector(names.weight, out, in); err != nil {
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

func loadNorm(m *Model, names decoderNormNames, size int, dst *decoderNorm) error {
	var err error
	if dst.weight, err = checkedTensor(m, names.weight, size); err != nil {
		return err
	}
	if dst.bias, err = checkedTensor(m, names.bias, size); err != nil {
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

func linearInto(dst, x []float32, l *decoderLinear, in, out int) {
	var err error
	if l.packed != nil {
		err = l.packed.Mul(dst[:out], x[:in])
	} else {
		err = whispergemm.MulVector(dst, l.weight, in, x[:in], out)
	}
	if err != nil {
		panic(err) // Internal callers use validated, fixed model dimensions.
	}
	bias := l.bias
	if bias != nil {
		for o := 0; o < out; o++ {
			dst[o] += bias[o]
		}
	}
}

// layerNormInto normalizes x into dst, using the NEON kernel when the row
// qualifies. Without acceleration it keeps the scalar float32 order exactly.
func layerNormInto(dst, x, weight, bias []float32) {
	n := len(x)
	if nn.Accelerated && n > 0 && n%8 == 0 && len(dst) >= n && len(weight) >= n && len(bias) >= n {
		nn.LayerNorm(x[:n], dst, weight, bias)
		return
	}
	layerNorm(dst, x, weight, bias)
}

// residualNormInto fuses x += projected with the following LayerNorm(x),
// matching the scalar addition order. Without the NEON kernel it performs
// the same two operations separately.
func residualNormInto(x, dst, projected, weight, bias []float32) {
	n := len(x)
	if nn.Accelerated && n > 0 && n%8 == 0 && len(dst) >= n && len(projected) >= n && len(weight) >= n && len(bias) >= n {
		nn.ResidualNorm(x, dst, weight, bias, projected, nil)
		return
	}
	addInto(x, projected)
	layerNorm(dst, x, weight, bias)
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
func (s *decoderScratch) attend(keys, values []float32, frames, valueStride int) error {
	heads := s.dims.TextHeads
	headSize := s.dims.TextState / heads
	scale := float32(math.Pow(float64(headSize), -0.25))
	for d, value := range s.query {
		s.scaledQuery[d] = value * scale
	}
	s.attention = decoderAttentionOperation{keyVec: s.layerKeys, valueVec: s.layerValues, heads: heads, dst: s.context, scaledQuery: s.scaledQuery, keys: keys, values: values, scores: s.scores, frames: frames, valueStride: valueStride}
	var err error
	if s.gemm != nil && frames >= 128 {
		err = s.gemm.Rows(&s.attention, heads, 1)
	} else {
		s.attention.ApplyRows(0, heads)
	}
	s.attention = decoderAttentionOperation{}
	return err
}

// attentionPackedInto is attentionCachedInto over per-head packed caches:
// scores, the NEON exponential, and the value product scaled by 1/sum.
func attentionPackedInto(dst, scaledQuery []float32, keys, values []*whispergemm.PackedVector, frames, heads int, scores []float32, firstHead, lastHead int) {
	headSize := len(scaledQuery) / heads
	for head := firstHead; head < lastHead; head++ {
		probabilities := scores[head*audioFrames : head*audioFrames+frames]
		start, end := head*headSize, (head+1)*headSize
		if err := keys[head].Mul(probabilities, scaledQuery[start:end]); err != nil {
			panic(err)
		}
		inverse := nn.SoftmaxExp(probabilities)
		out := dst[start:end]
		if err := values[head].Mul(out, probabilities); err != nil {
			panic(err)
		}
		for i := range out {
			out[i] *= inverse
		}
	}
}

func attentionCachedInto(dst, scaledQuery, keys, values []float32, frames, valueStride, heads int, scores []float32, firstHead, lastHead int) {
	state := len(scaledQuery)
	headSize := state / heads
	for head := firstHead; head < lastHead; head++ {
		probabilities := scores[head*audioFrames : head*audioFrames+frames]
		start, end := head*headSize, (head+1)*headSize
		if err := whispergemm.MulVector(probabilities, keys[start:], state, scaledQuery[start:end], frames); err != nil {
			panic(err)
		}
		if frames == 0 {
			clear(dst[start:end])
			continue
		}
		inverse := nn.SoftmaxExp(probabilities)
		out := dst[start:end]
		if err := whispergemm.MulVector(out, values[start*valueStride:], valueStride, probabilities, headSize); err != nil {
			panic(err)
		}
		for i := range out {
			out[i] *= inverse
		}
	}
}

// geluExactInto selects PyTorch's erf-based GELU curve. The shared scalar/SIMD
// evaluator has the FP32 numerical bound documented with internal/nn.GELU.
func geluExactInto(x []float32) {
	nn.BiasGELU(x, nil, 1, len(x))
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
