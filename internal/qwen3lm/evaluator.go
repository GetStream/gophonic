// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"errors"
	"fmt"
	"math"
	"runtime"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

// finite32 reports whether v is neither NaN nor infinite.
func finite32(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

const (
	// gemmAttentionMin is the sequence length from which attention runs as
	// blocked matrix products instead of the per-row streaming loop.
	gemmAttentionMin = 64
	// attentionBlock is the query rows per blocked-attention item.
	attentionBlock = 128
)

// attentionItem is one query block of one sequence, for one KV-head group
// (no prefix) or one query head (prefix modes, where group holds the head).
// Rows start at batch row start; past earlier keys come from a PrefixKV.
type attentionItem struct {
	start, q0, q1, group, past int32
}

// PrefixKV holds one token sequence's post-RoPE keys and values for every
// layer, so a later sequence that extends it (a growing conversation state)
// only evaluates its new tokens. Values are FP32, exactly as computed, so an
// extension yields the same result as evaluating the whole sequence up to
// floating-point reassociation. It belongs to one Evaluator; it is not safe for
// concurrent use.
type PrefixKV struct {
	owner        *Evaluator
	tokens       []int
	keys, values [][]float32 // per layer, [capacity][kvDim]
	capacity     int
	packs        []prefixPack // per layer: the same keys and values packed for attention
	gpu          *gpuPrefix   // GPU models keep keys and values in GPU memory instead
}

// prefixPack keeps one layer's stored keys and values packed per KV group,
// so attention over a long prefix packs only tokens added since the last
// call. Keys are packed transposed (positions are output columns, which can
// be appended); values are packed in chunks of prefixChunk positions.
type prefixPack struct {
	keysT  []*whispergemm.PackedB   // per group: K=headDim, N=packed positions
	values [][]*whispergemm.PackedB // per group, per chunk: K≤prefixChunk positions, N=headDim
	n      []int                    // per group: positions packed
}

// prefixChunk is the number of positions per packed value chunk.
const prefixChunk = 256

// NewPrefixKV allocates storage for up to capacity tokens:
// 8 bytes × layers × KV width per token (288 KiB for Qwen3-8B).
func (e *Evaluator) NewPrefixKV(capacity int) (*PrefixKV, error) {
	if e == nil || e.m == nil {
		return nil, errors.New("qwen3: nil evaluator")
	}
	c := &e.m.cfg
	limit := c.maxPositions
	if e.m.gpu != nil {
		limit = e.m.gpu.maxPositions()
	}
	if capacity < 1 || capacity > limit {
		return nil, fmt.Errorf("qwen3: prefix capacity %d outside [1,%d]", capacity, limit)
	}
	if e.m.gpu != nil {
		pre, err := e.m.gpu.newPrefix(capacity)
		if err != nil {
			return nil, err
		}
		kv := &PrefixKV{owner: e, tokens: make([]int, 0, capacity), capacity: capacity, gpu: pre}
		runtime.AddCleanup(kv, func(p *gpuPrefix) { p.release() }, pre)
		return kv, nil
	}
	kv := &PrefixKV{owner: e, tokens: make([]int, 0, capacity), capacity: capacity,
		keys: make([][]float32, c.layers), values: make([][]float32, c.layers)}
	for l := range c.layers {
		kv.keys[l] = make([]float32, capacity*c.kvDim)
		kv.values[l] = make([]float32, capacity*c.kvDim)
	}
	return kv, nil
}

// CopyPrefix makes kv hold the first p tokens of src.
func (kv *PrefixKV) CopyPrefix(src *PrefixKV, p int) {
	c := &kv.owner.m.cfg
	n := p * c.kvDim
	if kv.gpu != nil {
		kv.gpu.copyFrom(src.gpu, c.layers, n)
	}
	for l := range kv.keys {
		copy(kv.keys[l][:n], src.keys[l][:n])
		copy(kv.values[l][:n], src.values[l][:n])
	}
	kv.tokens = append(kv.tokens[:0], src.tokens[:p]...)
}

// Capacity reports the most tokens kv can hold.
func (kv *PrefixKV) Capacity() int { return kv.capacity }

// Tokens returns the token sequence whose keys and values kv holds. The slice
// is owned by kv and changes on the next extension.
func (kv *PrefixKV) Tokens() []int { return kv.tokens }

// CommonPrefix returns how many leading tokens of ids kv already holds.
func (kv *PrefixKV) CommonPrefix(ids []int) int {
	n := 0
	for n < len(ids) && n < len(kv.tokens) && ids[n] == kv.tokens[n] {
		n++
	}
	return n
}

// HiddenLastExtendInto evaluates ids as the continuation of the first keep
// tokens held in kv and writes the last token's post-final-RMSNorm state to
// dst. Only len(ids) tokens are computed; they attend to the kept prefix.
// Afterwards kv holds kv.Tokens()[:keep] followed by ids. keep may be zero.
func (e *Evaluator) HiddenLastExtendInto(kv *PrefixKV, keep int, ids []int, dst []float32, ws *Workspace) error {
	return e.HiddenLastExtendEmbedInto(kv, keep, ids, Embeds{}, dst, ws)
}

// Embeds replaces the input embeddings of a placeholder token, as a
// multimodal model splices encoder outputs into its prompt (Qwen3-ASR's audio
// features at <|audio_pad|>): the i-th occurrence of Token takes row i of
// Rows, which holds one hidden-wide row per occurrence. The zero value
// replaces nothing.
type Embeds struct {
	Token int
	Rows  []float32
}

// HiddenLastExtendEmbedInto is HiddenLastExtendInto with the placeholder
// rows of embeds. kv records each replaced token as -1, so CommonPrefix never
// matches a later sequence across different embeddings.
func (e *Evaluator) HiddenLastExtendEmbedInto(kv *PrefixKV, keep int, ids []int, embeds Embeds, dst []float32, ws *Workspace) error {
	if kv == nil || kv.owner != e {
		return errors.New("qwen3: prefix store belongs to another evaluator")
	}
	if keep < 0 || keep > len(kv.tokens) {
		return fmt.Errorf("qwen3: keep %d outside the %d stored tokens", keep, len(kv.tokens))
	}
	if keep+len(ids) > kv.capacity {
		return fmt.Errorf("qwen3: %d tokens exceed prefix capacity %d", keep+len(ids), kv.capacity)
	}
	if ws == nil {
		return errors.New("qwen3: nil workspace")
	}
	if len(embeds.Rows) != 0 || e.m.embed == nil {
		n := 0
		for _, id := range ids {
			if id == embeds.Token {
				n++
			}
		}
		if e.m.embed == nil && (n != len(ids) || len(embeds.Rows) == 0) {
			return errors.New("qwen3: weights loaded without an embedding table take only Embeds rows")
		}
		if len(embeds.Rows) != n*e.m.cfg.hidden {
			return fmt.Errorf("qwen3: %d embedding values for %d placeholders of width %d", len(embeds.Rows), n, e.m.cfg.hidden)
		}
		ws.embeds = embeds
	}
	kv.tokens = kv.tokens[:keep] // the stored suffix is overwritten below
	ws.prefix, ws.past = kv, keep
	ws.oneSeq[0], ws.oneDst[0] = ids, dst
	err := e.HiddenLastBatchInto(ws.oneSeq[:], ws.oneDst[:], ws)
	ws.oneSeq[0], ws.oneDst[0] = nil, nil
	ws.prefix, ws.past, ws.embeds = nil, 0, Embeds{}
	if err != nil {
		return err
	}
	for _, id := range ids {
		if len(embeds.Rows) != 0 && id == embeds.Token {
			id = -1
		}
		kv.tokens = append(kv.tokens, id)
	}
	return nil
}

// maxTail bounds the states of HiddenTailExtendEmbedInto and
// LogitsRowsInto, within one GPU pass.
const maxTail = 256

// HiddenTailExtendEmbedInto is HiddenLastExtendEmbedInto that writes the
// post-final-RMSNorm states of the last len(dst)/hidden positions to dst, in
// order. With LogitsRowsInto it checks a draft continuation in one pass: the
// state at each position predicts the token after it.
func (e *Evaluator) HiddenTailExtendEmbedInto(kv *PrefixKV, keep int, ids []int, embeds Embeds, dst []float32, ws *Workspace) error {
	if e == nil || e.m == nil || ws == nil {
		return errors.New("qwen3: nil evaluator or workspace")
	}
	h := e.m.cfg.hidden
	k := len(dst) / h
	if k == 0 || len(dst) != k*h || k > len(ids) || k > maxTail {
		return fmt.Errorf("qwen3: %d tail values for %d tokens of width %d", len(dst), len(ids), h)
	}
	ws.tail = dst
	defer func() { ws.tail = nil }()
	return e.HiddenLastExtendEmbedInto(kv, keep, ids, embeds, dst[(k-1)*h:], ws)
}

// LogitsRowsInto is LogitsInto for several states: hidden holds whole
// states, and dst receives the logits of each in turn.
func (e *Evaluator) LogitsRowsInto(hidden, dst []float32, ws *Workspace) error {
	if e == nil || e.m == nil || ws == nil || ws.owner != e {
		return errors.New("qwen3: nil evaluator or foreign workspace")
	}
	c := &e.m.cfg
	k := len(hidden) / c.hidden
	if k == 0 || len(hidden) != k*c.hidden || len(dst) != k*c.vocab || k > maxTail {
		return fmt.Errorf("qwen3: logits of %d values into %d, want whole states of %d and %d", len(hidden), len(dst), c.hidden, c.vocab)
	}
	if ws.gpu != nil && k > 1 {
		if e.m.gpu == nil || !e.m.gpu.hasHead() {
			return errors.New("qwen3: the weights were loaded without a language-model head")
		}
		return ws.gpu.logitsRowsInto(e.m, hidden, dst, k)
	}
	for r := range k {
		if err := e.LogitsInto(hidden[r*c.hidden:(r+1)*c.hidden], dst[r*c.vocab:(r+1)*c.vocab], ws); err != nil {
			return err
		}
	}
	return nil
}

// LogitsInto writes the language-model head's logits for one
// post-final-RMSNorm hidden state (a HiddenLast result) to dst, which has one
// value per vocabulary entry. The weights must have been loaded with
// LoadOptions.Head. Warm calls allocate nothing.
func (e *Evaluator) LogitsInto(hidden, dst []float32, ws *Workspace) error {
	if e == nil || e.m == nil || ws == nil || ws.owner != e {
		return errors.New("qwen3: nil evaluator or foreign workspace")
	}
	m, c := e.m, &e.m.cfg
	if m.head.f16 == nil && m.head.i8 == nil && (m.gpu == nil || !m.gpu.hasHead()) {
		return errors.New("qwen3: the weights were loaded without a language-model head")
	}
	if len(hidden) != c.hidden || len(dst) != c.vocab {
		return fmt.Errorf("qwen3: logits of a %d-wide state into %d values, want %d and %d", len(hidden), len(dst), c.hidden, c.vocab)
	}
	if ws.gpu != nil {
		return ws.gpu.logitsInto(m, hidden, dst)
	}
	if err := ws.ensure(c, 1, 1); err != nil {
		return err
	}
	ws.pool.hold()
	defer ws.pool.release()
	copy(ws.norm, hidden)
	ws.op.rows = 1
	ws.project(ws.norm, c.hidden, false, projection{&m.head, dst})
	return nil
}

// attentionScratch is one participant's blocked-attention storage.
type attentionScratch struct {
	keysT, values *whispergemm.PackedB
	scores        []float32
	gemm          []float32 // whispergemm.MulScratch scratch
	tmp           []float32 // one P·V chunk product, [attentionBlock][headDim]
}

// Evaluator computes Qwen3 last-token hidden states a layer at a time. All
// tokens of a call, including tokens from several independent sequences, pass
// through each projection together, so each packed weight tile is read once
// per 16 tokens instead of once per token or per sequence.
type Evaluator struct {
	m *Weights
}

// NewEvaluator returns an evaluator over m. It holds no mutable state; create
// one Workspace per concurrent call.
func NewEvaluator(m *Weights) (*Evaluator, error) {
	if m == nil {
		return nil, errors.New("qwen3: nil Qwen3 model")
	}
	return &Evaluator{m: m}, nil
}

// HiddenLastSharedInto evaluates independent sequences that each continue
// the tokens held in kv, and writes each one's last-token state to dst. kv is
// read, never written, so one prefix (for example a prepared prompt) can
// serve any number of batches. Every sequence attends to the whole prefix and
// to its own earlier tokens.
func (e *Evaluator) HiddenLastSharedInto(kv *PrefixKV, seqs [][]int, dst [][]float32, ws *Workspace) error {
	if kv == nil || kv.owner != e {
		return errors.New("qwen3: prefix store belongs to another evaluator")
	}
	if ws == nil {
		return errors.New("qwen3: nil workspace")
	}
	ws.prefix, ws.shared, ws.past = kv, true, len(kv.tokens)
	err := e.HiddenLastBatchInto(seqs, dst, ws)
	ws.prefix, ws.shared, ws.past = nil, false, 0
	return err
}

// Workspace owns reusable activations for a packed batch of rows, one layer's
// keys and values, per-tile activation scratch, and optionally a pool of
// persistent workers. It belongs to one evaluator and one concurrent call.
type Workspace struct {
	h, norm, q, ctx, attn, gate, up []float32
	keys, values                    []float32 // K and V projections of the current layer, [rows][kvDim]
	rowStart, rowPos, rowLen        []int32   // each row's sequence start, position, and sequence length
	attnItems                       []attentionItem
	attnScratch                     []attentionScratch // per participant
	prefix                          *PrefixKV          // set by HiddenLastExtendInto and HiddenLastSharedInto
	embeds                          Embeds             // set by HiddenLastExtendEmbedInto
	tail                            []float32          // set by HiddenTailExtendEmbedInto
	shared                          bool               // prefix is read-only and shared by every sequence
	attnPerHead                     bool               // prefix attention items are per query head, not per group
	past                            int
	ropeCos, ropeSin                []float32
	ropePositions                   int // positions whose RoPE values are filled
	positions                       int // position capacity of RoPE and attention scratch
	gpu                             *gpuWorkspace
	capacity                        int
	owner                           *Evaluator
	tiles                           []*q8gemm.Workspace
	tilesI8                         []*q8gemm.WorkspaceI8 // int8 weights only
	rotated                         []float32             // rotated inputs, int8 weights only
	scratch                         []*q8gemm.Scratch     // per participant; nil entries with SME
	pool                            *workerPool
	op                              layerOp
	oneSeq                          [1][]int
	oneDst                          [1][]float32
}

// NewWorkspace allocates a reusable workspace whose projections, attention,
// and elementwise stages are split across workers goroutines, including the
// caller; workers must be between 1 and 64. The workspace starts workers-1
// persistent goroutines; call Close to release them. Activation buffers grow
// on the first call for a new maximum batch size; later calls up to that size
// allocate nothing.
func (e *Evaluator) NewWorkspace(workers int) (*Workspace, error) {
	if e == nil || e.m == nil {
		return nil, errors.New("qwen3: nil evaluator")
	}
	if workers < 1 || workers > 64 {
		return nil, fmt.Errorf("qwen3: invalid worker count %d", workers)
	}
	ws := &Workspace{owner: e}
	ws.op.ws = ws
	if e.m.gpu != nil {
		var err error
		if ws.gpu, err = e.m.gpu.newWorkspace(); err != nil {
			return nil, err
		}
		return ws, nil
	}
	if workers > 1 {
		ws.pool = newWorkerPool(workers)
	}
	c := &e.m.cfg
	ws.attnScratch = make([]attentionScratch, workers)
	ws.scratch = make([]*q8gemm.Scratch, workers)
	for i := range ws.scratch {
		ws.scratch[i] = q8gemm.NewScratch(max(c.hidden, c.heads*c.headDim, c.intermediate))
	}
	return ws, nil
}

// Close stops the workspace's worker goroutines. The workspace must not be
// used afterwards.
func (ws *Workspace) Close() error {
	if ws != nil && ws.gpu != nil {
		ws.gpu.release()
		ws.gpu = nil
	}
	if ws == nil || ws.pool == nil {
		return nil
	}
	ws.pool.close()
	ws.pool = nil
	return nil
}

// HiddenLastInto evaluates ids with causal attention and writes the
// post-final-RMSNorm representation of the last token to dst.
func (e *Evaluator) HiddenLastInto(ids []int, dst []float32, ws *Workspace) error {
	if ws == nil {
		return errors.New("qwen3: nil workspace")
	}
	ws.oneSeq[0], ws.oneDst[0] = ids, dst
	err := e.HiddenLastBatchInto(ws.oneSeq[:], ws.oneDst[:], ws)
	ws.oneSeq[0], ws.oneDst[0] = nil, nil
	return err
}

// HiddenLastBatchInto evaluates independent sequences in one packed forward
// pass and writes each sequence's post-final-RMSNorm last-token vector to the
// matching dst row. Each sequence attends only to its own earlier tokens and
// starts at position zero, so every result equals a separate evaluation up to
// floating-point reassociation. After the workspace has seen this total token
// count, the method allocates no heap memory.
func (e *Evaluator) HiddenLastBatchInto(seqs [][]int, dst [][]float32, ws *Workspace) error {
	if e == nil || e.m == nil || ws == nil {
		return errors.New("qwen3: nil evaluator or workspace")
	}
	if ws.owner != e {
		return errors.New("qwen3: workspace belongs to another evaluator")
	}
	m, c := e.m, &e.m.cfg
	if len(seqs) == 0 || len(dst) != len(seqs) {
		return fmt.Errorf("qwen3: %d destinations for %d sequences", len(dst), len(seqs))
	}
	rows, longest := 0, 0
	for s, ids := range seqs {
		if len(ids) == 0 {
			return errors.New("qwen3: empty token sequence")
		}
		if ws.past+len(ids) > c.maxPositions {
			return fmt.Errorf("qwen3: %d tokens exceeds Qwen3 context %d", ws.past+len(ids), c.maxPositions)
		}
		if len(dst[s]) != c.hidden {
			return fmt.Errorf("qwen3: destination width %d, want %d", len(dst[s]), c.hidden)
		}
		for i, id := range ids {
			if id < 0 || id >= c.vocab {
				return fmt.Errorf("qwen3: token %d at position %d is outside vocabulary", id, i)
			}
		}
		if rows > int(^uint(0)>>2)-len(ids) {
			return errors.New("qwen3: batch size overflows int")
		}
		rows += len(ids)
		longest = max(longest, len(ids))
	}
	if ws.prefix != nil && !ws.shared && len(seqs) != 1 {
		return errors.New("qwen3: a prefix extension evaluates exactly one sequence")
	}
	if ws.gpu != nil {
		var pre *gpuPrefix
		if ws.prefix != nil {
			pre = ws.prefix.gpu
		}
		ws.gpu.tail = ws.tail
		defer func() { ws.gpu.tail = nil }()
		return ws.gpu.batch(m, seqs, dst, pre, ws.past, ws.shared, ws.embeds)
	}
	if err := ws.ensure(c, rows, ws.past+longest); err != nil {
		return err
	}
	ws.pool.hold()
	defer ws.pool.release()
	row := 0
	ws.attnItems = ws.attnItems[:0]
	for _, ids := range seqs {
		start := int32(row)
		// With a prefix, every row takes the blocked path, which reads the
		// kept keys; the streaming loop sees an out-of-range length and skips.
		seqLen := int32(len(ids))
		if ws.prefix != nil {
			seqLen = math.MaxInt32
		}
		spliced := 0
		for pos, id := range ids {
			if h := ws.h[row*c.hidden : (row+1)*c.hidden]; len(ws.embeds.Rows) != 0 && id == ws.embeds.Token {
				copy(h, ws.embeds.Rows[spliced*c.hidden:])
				spliced++
			} else {
				m.embedRow(id, h)
			}
			ws.rowStart[row], ws.rowPos[row], ws.rowLen[row] = start, int32(ws.past+pos), seqLen
			row++
		}
		if len(ids) >= gemmAttentionMin || ws.prefix != nil {
			// Later query blocks attend to more keys; queue them first so the
			// dynamic scheduler finishes with the cheapest items.
			for q0 := (len(ids) - 1) / attentionBlock * attentionBlock; q0 >= 0; q0 -= attentionBlock {
				for g := range c.kvHeads {
					ws.attnItems = append(ws.attnItems, attentionItem{start, int32(q0), int32(min(q0+attentionBlock, len(ids))), int32(g), int32(ws.past)})
				}
			}
		}
	}
	ws.prepareRoPE(c, ws.past+longest)
	// With a prefix and few items (one long prefix, few new rows), split
	// items per query head so every worker gets work; with many items the
	// per-group form shares each group's packed rows across its heads.
	ws.attnPerHead = false
	if ws.prefix != nil && len(ws.attnItems) < 4*ws.pool.size() {
		ws.attnPerHead = true
		group := c.heads / c.kvHeads
		n := len(ws.attnItems)
		for i := range n {
			it := ws.attnItems[i]
			for h := 1; h < group; h++ {
				it2 := it
				it2.group = it.group*int32(group) + int32(h)
				ws.attnItems = append(ws.attnItems, it2)
			}
			ws.attnItems[i].group = it.group * int32(group)
		}
	}

	op := &ws.op
	op.rows = rows
	for layer := range m.layers {
		l := &m.layers[layer]
		op.layer, op.layerIndex = l, layer
		// The previous layer's MLP output waits in attn; fold that residual
		// add into this layer's attention RMSNorm.
		residual := ws.attn
		if layer == 0 {
			residual = nil
		}
		ws.addNorm(residual, l.attnNorm, &l.q)
		ws.project(ws.norm, c.hidden, true, projection{&l.q, ws.q}, projection{&l.k, ws.keys}, projection{&l.v, ws.values})
		ws.run(opQKRope, rows, 4)
		if kv := ws.prefix; kv != nil && !ws.shared {
			// Store this layer's new keys and values after the kept prefix;
			// blocked attention then reads the whole causal range from kv.
			n := rows * c.kvDim
			copy(kv.keys[layer][ws.past*c.kvDim:ws.past*c.kvDim+n], ws.keys[:n])
			copy(kv.values[layer][ws.past*c.kvDim:ws.past*c.kvDim+n], ws.values[:n])
		}
		ws.run(opAttention, rows*c.kvHeads, 4)
		if len(ws.attnItems) > 0 {
			if ws.prefix != nil {
				// Pack any stored prefix tokens not yet packed for this
				// layer, then attend to the packed prefix plus own rows.
				ws.preparePrefixPack(layer)
				ws.run(opPackPrefix, c.kvHeads, 1)
				ws.run(opAttentionPrefix, len(ws.attnItems), 1)
			} else {
				ws.run(opAttentionGEMM, len(ws.attnItems), 1)
			}
		}
		if layer == len(m.layers)-1 {
			// Only each sequence's last row (or a tail) reaches the output,
			// so the final output projection and MLP run on those rows alone.
			if k := len(ws.tail) / c.hidden; k > 1 {
				ws.keepTailRows(rows, k, c)
			} else {
				ws.keepLastRows(seqs, c)
			}
		}
		ws.project(ws.ctx, c.heads*c.headDim, false, projection{&l.o, ws.attn})
		ws.addNorm(ws.attn, l.mlpNorm, &l.gate)
		ws.project(ws.norm, c.hidden, true, projection{&l.gate, ws.gate}, projection{&l.up, ws.up})
		ws.run(opSwiGLU, op.rows*swigluChunks(c.intermediate), 4)
		ws.project(ws.gate, c.intermediate, false, projection{&l.down, ws.attn})
	}
	op.layer = nil
	if k := len(ws.tail) / c.hidden; k > 1 {
		for r := range k {
			last := ws.h[r*c.hidden : (r+1)*c.hidden]
			addInto(last, ws.attn[r*c.hidden:(r+1)*c.hidden])
			out := ws.tail[r*c.hidden : (r+1)*c.hidden]
			rmsNorm32(out, last, m.finalNorm, c.eps)
			for _, value := range out {
				if !finite32(value) {
					return errors.New("qwen3: non-finite hidden state")
				}
			}
		}
		return nil
	}
	for s := range seqs {
		last := ws.h[s*c.hidden : (s+1)*c.hidden]
		addInto(last, ws.attn[s*c.hidden:(s+1)*c.hidden])
		rmsNorm32(dst[s], last, m.finalNorm, c.eps)
		for _, value := range dst[s] {
			if !finite32(value) {
				return errors.New("qwen3: non-finite hidden state")
			}
		}
	}
	return nil
}

// keepLastRows moves each sequence's last row of h and ctx to row s and
// shrinks the batch to one row per sequence. Rows only move toward the
// front, so copying in sequence order never overwrites a row still needed.
func (ws *Workspace) keepLastRows(seqs [][]int, c *modelConfig) {
	qdim := c.heads * c.headDim
	row := 0
	for s, ids := range seqs {
		row += len(ids)
		if last := row - 1; last != s {
			copy(ws.h[s*c.hidden:(s+1)*c.hidden], ws.h[last*c.hidden:(last+1)*c.hidden])
			copy(ws.ctx[s*qdim:(s+1)*qdim], ws.ctx[last*qdim:(last+1)*qdim])
		}
	}
	ws.op.rows = len(seqs)
}

// keepTailRows moves the last k of rows rows of h and ctx to the front and
// shrinks the batch to them.
func (ws *Workspace) keepTailRows(rows, k int, c *modelConfig) {
	qdim := c.heads * c.headDim
	copy(ws.h[:k*c.hidden], ws.h[(rows-k)*c.hidden:rows*c.hidden])
	copy(ws.ctx[:k*qdim], ws.ctx[(rows-k)*qdim:rows*qdim])
	ws.op.rows = k
}

// preparePrefixPack allocates the per-group slices of the prefix's packed
// copy for layer before the parallel pack stage fills them.
func (ws *Workspace) preparePrefixPack(layer int) {
	kv, c := ws.prefix, &ws.owner.m.cfg
	if kv.packs == nil {
		kv.packs = make([]prefixPack, c.layers)
	}
	if pk := &kv.packs[layer]; pk.keysT == nil {
		pk.keysT = make([]*whispergemm.PackedB, c.kvHeads)
		pk.values = make([][]*whispergemm.PackedB, c.kvHeads)
		pk.n = make([]int, c.kvHeads)
	}
}

// Reserve allocates storage for batches of up to rows tokens and sequences
// of up to positions tokens (including any kept prefix), so later calls
// within those bounds never allocate. Storage otherwise grows on demand by
// doubling.
func (ws *Workspace) Reserve(rows, positions int) error {
	if ws == nil || ws.owner == nil {
		return errors.New("qwen3: nil workspace")
	}
	c := &ws.owner.m.cfg
	if rows < 1 || positions < 1 || positions > c.maxPositions {
		return fmt.Errorf("qwen3: cannot reserve %d rows and %d positions", rows, positions)
	}
	if ws.gpu != nil {
		return nil // GPU buffers are allocated whole
	}
	if err := ws.ensure(c, rows, positions); err != nil {
		return err
	}
	ws.prepareRoPE(c, positions)
	return nil
}

func (ws *Workspace) ensure(c *modelConfig, n, positions int) error {
	if ws.capacity < n {
		capN := max(16, ws.capacity)
		for capN < n {
			capN *= 2
		}
		qdim := c.heads * c.headDim
		maxInt := int(^uint(0) >> 1)
		for _, dim := range [...]int{c.hidden, qdim, c.kvDim, c.intermediate} {
			if capN > maxInt/dim {
				return errors.New("qwen3: workspace size overflows int")
			}
		}
		ws.h = make([]float32, capN*c.hidden)
		ws.norm = make([]float32, capN*c.hidden)
		ws.q = make([]float32, capN*qdim)
		ws.ctx = make([]float32, capN*qdim)
		ws.attn = make([]float32, capN*c.hidden)
		ws.gate = make([]float32, capN*c.intermediate)
		ws.up = make([]float32, capN*c.intermediate)
		ws.keys = make([]float32, capN*c.kvDim)
		ws.values = make([]float32, capN*c.kvDim)
		ws.rowStart = make([]int32, capN)
		ws.rowPos = make([]int32, capN)
		ws.rowLen = make([]int32, capN)
		tiles := (capN + q8gemm.ActivationRows - 1) / q8gemm.ActivationRows
		widest := max(c.hidden, qdim, c.intermediate)
		if ws.owner.m.format == WeightsInt8 {
			ws.rotated = make([]float32, capN*widest)
			for len(ws.tilesI8) < tiles {
				tile, err := q8gemm.NewWorkspaceI8(widest)
				if err != nil {
					return fmt.Errorf("qwen3: activation tile: %w", err)
				}
				ws.tilesI8 = append(ws.tilesI8, tile)
			}
		} else {
			for len(ws.tiles) < tiles {
				tile, err := q8gemm.NewWorkspace(widest)
				if err != nil {
					return fmt.Errorf("qwen3: activation tile: %w", err)
				}
				ws.tiles = append(ws.tiles, tile)
			}
		}
		ws.capacity = capN
	}
	if positions > ws.positions {
		// Position-sized storage doubles, so it is reallocated at most
		// log2(maxPositions) times over a workspace's life.
		capP := max(16, ws.positions)
		for capP < positions {
			capP *= 2
		}
		capP = min(capP, c.maxPositions)
		ws.ropeCos = make([]float32, capP*c.headDim/2)
		ws.ropeSin = make([]float32, capP*c.headDim/2)
		ws.ropePositions = 0
		// Prefix modes always use blocked attention, so the scratch exists
		// for every capacity.
		for i := range ws.attnScratch {
			sc := &ws.attnScratch[i]
			var err error
			if sc.keysT, err = whispergemm.NewPackedB(c.headDim, capP); err != nil {
				return err
			}
			if sc.values, err = whispergemm.NewPackedB(capP, c.headDim); err != nil {
				return err
			}
			sc.scores = make([]float32, attentionBlock*capP)
			sc.gemm = make([]float32, whispergemm.ScratchLen(max(capP, c.headDim)))
			sc.tmp = make([]float32, attentionBlock*c.headDim)
		}
		ws.positions = capP
	}
	return nil
}

// prepareRoPE extends the position-only rotation tables. They never change for
// a given model, so warmed calls only compute positions not seen before.
func (ws *Workspace) prepareRoPE(c *modelConfig, n int) {
	half := c.headDim / 2
	for pos := ws.ropePositions; pos < n; pos++ {
		base := pos * half
		for d, inv := range c.invFreq {
			theta := float64(pos) * inv
			ws.ropeCos[base+d] = float32(math.Cos(theta))
			ws.ropeSin[base+d] = float32(math.Sin(theta))
		}
	}
	ws.ropePositions = max(ws.ropePositions, n)
}

// run applies the current op kind to items, splitting across workers when the
// workspace has them; each claim covers grain items. The op fields must be set
// before the call.
func (ws *Workspace) run(kind opKind, items, grain int) {
	ws.op.kind = kind
	if ws.pool == nil {
		ws.op.ApplyRows(0, 0, items)
		return
	}
	// Ordinary stages use at most maxSMEWorkers participants: waiting on
	// more costs more at each barrier than their extra throughput gains.
	ws.pool.runN(&ws.op, items, grain, maxSMEWorkers)
}

// runEach runs the current op kind once on every participant.
func (ws *Workspace) runEach(kind opKind, n int) {
	ws.op.kind = kind
	if ws.pool == nil {
		ws.op.ApplyRows(0, 0, 1)
		return
	}
	ws.pool.runEach(&ws.op, n)
}

const (
	maxSMEWorkers = 8 // streaming SME saturates both P-cluster units near here
	coexecSME     = 4 // SME workers when NEON strips co-execute
	coexecTiles   = 4 // minimum 16-row tiles for co-execution to pay
)

// addNorm computes h += residual (when residual is non-nil) and
// norm = RMSNorm(h) * weight for every row. With next set, the same row pass
// also prepares next's activation tiles: it rotates each normalized row for
// int8 weights and sets the row's quantization scale, so the following
// project call skips its own rotate and row-scale stages.
func (ws *Workspace) addNorm(residual, weight []float32, next *linear) {
	op := &ws.op
	op.residual, op.normWeight, op.scaleRows = residual, weight, next != nil
	if next != nil {
		op.rot = next.rot
		ws.prepareTiles(ws.owner.m.cfg.hidden)
	}
	ws.run(opAddNorm, op.rows, 2)
	op.residual, op.normWeight, op.scaleRows, op.rot = nil, nil, false, nil
}

// prepareTiles records the next projection's tile shapes; op.rot selects the
// int8 tiles.
func (ws *Workspace) prepareTiles(cols int) {
	rows := ws.op.rows
	tiles := (rows + q8gemm.ActivationRows - 1) / q8gemm.ActivationRows
	for t := range tiles {
		n := min(q8gemm.ActivationRows, rows-t*q8gemm.ActivationRows)
		var err error
		if ws.op.rot != nil {
			err = ws.tilesI8[t].Prepare(n, cols)
		} else {
			err = ws.tiles[t].Prepare(n, cols)
		}
		if err != nil {
			panic("qwen3: activation tile: " + err.Error())
		}
	}
}

type projection struct {
	l   *linear
	dst []float32
}

// project multiplies the rows of src (width cols) by up to three projections
// that share that input. Each 16-row tile is scaled and converted once, then
// every panel of every projection runs in one parallel dispatch. Shapes are
// validated when the model loads.
func (ws *Workspace) project(src []float32, cols int, prepared bool, projs ...projection) {
	op := &ws.op
	rows := op.rows
	op.panels = 0
	for i, p := range projs {
		op.proj[i] = p
		op.panelEnd[i] = op.panels + p.l.panels()
		op.panels = op.panelEnd[i]
	}
	// Projections sharing an input share a format and rotation.
	op.rot = projs[0].l.rot
	tiles := (rows + q8gemm.ActivationRows - 1) / q8gemm.ActivationRows
	op.src, op.cols = src, cols
	if op.rot != nil {
		op.src = ws.rotated
	}
	if !prepared {
		ws.prepareTiles(cols)
		op.src = src
		ws.run(opRowScale, rows, 2)
		if op.rot != nil {
			op.src = ws.rotated
		}
	}
	ws.run(opPack, tiles*packChunks(cols), 1)
	// SME saturates near eight streaming threads. With int8 weights and at
	// least four tiles, four SME workers plus NEON strips on the remaining
	// cores are faster; with fewer tiles NEON only adds contention.
	size := ws.pool.size()
	op.coexec = op.rot != nil && tiles >= coexecTiles && size > coexecSME
	op.smeWorkers = min(size, maxSMEWorkers)
	if op.coexec {
		op.smeWorkers = coexecSME
	}
	op.claims.Store(uint64(op.panels*stripsPerPanel)<<32 | 0)
	participants := op.smeWorkers
	if op.coexec {
		participants = size
	}
	ws.runEach(opProject, participants)
	op.src, op.rot = nil, nil
	for i := range op.proj {
		op.proj[i] = projection{}
	}
}

// Dot32 returns the FP32 dot product of a and b[:len(a)] with the kernel the
// evaluator uses.
func Dot32(a, b []float32) float32 { return dot32(a, b) }
