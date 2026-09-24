// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"errors"
	"fmt"
	"math"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

const (
	// gemmAttentionMin is the sequence length from which attention runs as
	// blocked matrix products instead of the per-row streaming loop.
	gemmAttentionMin = 64
	// attentionBlock is the query rows per blocked-attention item.
	attentionBlock = 128
)

// attentionItem is one KV-head group and one query block of one sequence.
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
}

// NewPrefixKV allocates storage for up to capacity tokens:
// 8 bytes × layers × KV width per token (288 KiB for Qwen3-8B).
func (e *Evaluator) NewPrefixKV(capacity int) (*PrefixKV, error) {
	if e == nil || e.m == nil {
		return nil, errors.New("qwen3: nil evaluator")
	}
	c := &e.m.cfg
	if capacity < 1 || capacity > c.maxPositions {
		return nil, fmt.Errorf("qwen3: prefix capacity %d outside [1,%d]", capacity, c.maxPositions)
	}
	kv := &PrefixKV{owner: e, tokens: make([]int, 0, capacity), capacity: capacity,
		keys: make([][]float32, c.layers), values: make([][]float32, c.layers)}
	for l := range c.layers {
		kv.keys[l] = make([]float32, capacity*c.kvDim)
		kv.values[l] = make([]float32, capacity*c.kvDim)
	}
	return kv, nil
}

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
	kv.tokens = kv.tokens[:keep] // the stored suffix is overwritten below
	ws.prefix, ws.past = kv, keep
	ws.oneSeq[0], ws.oneDst[0] = ids, dst
	err := e.HiddenLastBatchInto(ws.oneSeq[:], ws.oneDst[:], ws)
	ws.oneSeq[0], ws.oneDst[0] = nil, nil
	ws.prefix, ws.past = nil, 0
	if err != nil {
		return err
	}
	kv.tokens = append(kv.tokens, ids...)
	return nil
}

// attentionScratch is one participant's blocked-attention storage.
type attentionScratch struct {
	keysT, values *whispergemm.PackedB
	scores        []float32
	gemm          []float32 // whispergemm.MulScratch scratch
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

// Workspace owns reusable activations for a packed batch of rows, one layer's
// keys and values, per-tile activation scratch, and optionally a pool of
// persistent workers. It belongs to one evaluator and one concurrent call.
type Workspace struct {
	h, norm, q, ctx, attn, gate, up []float32
	keys, values                    []float32 // K and V projections of the current layer, [rows][kvDim]
	rowStart, rowPos, rowLen        []int32   // each row's sequence start, position, and sequence length
	attnItems                       []attentionItem
	attnScratch                     []attentionScratch // per participant
	prefix                          *PrefixKV          // set only by HiddenLastExtendInto
	past                            int
	ropeCos, ropeSin                []float32
	ropePositions                   int // positions whose RoPE values are filled
	positions                       int // position capacity of RoPE and attention scratch
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
	if ws.prefix != nil && len(seqs) != 1 {
		return errors.New("qwen3: a prefix extension evaluates exactly one sequence")
	}
	if err := ws.ensure(c, rows, ws.past+longest); err != nil {
		return err
	}
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
		for pos, id := range ids {
			m.embedRow(id, ws.h[row*c.hidden:(row+1)*c.hidden])
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
		ws.addNorm(residual, l.attnNorm)
		ws.project(ws.norm, c.hidden, projection{&l.q, ws.q}, projection{&l.k, ws.keys}, projection{&l.v, ws.values})
		ws.run(opQKRope, rows, 4)
		if kv := ws.prefix; kv != nil {
			// Store this layer's new keys and values after the kept prefix;
			// blocked attention then reads the whole causal range from kv.
			n := rows * c.kvDim
			copy(kv.keys[layer][ws.past*c.kvDim:ws.past*c.kvDim+n], ws.keys[:n])
			copy(kv.values[layer][ws.past*c.kvDim:ws.past*c.kvDim+n], ws.values[:n])
		}
		ws.run(opAttention, rows*c.kvHeads, 4)
		if len(ws.attnItems) > 0 {
			ws.run(opAttentionGEMM, len(ws.attnItems), 1)
		}
		if layer == len(m.layers)-1 {
			// Only each sequence's last row reaches the output, so the final
			// output projection and MLP run on those rows alone.
			ws.keepLastRows(seqs, c)
		}
		ws.project(ws.ctx, c.heads*c.headDim, projection{&l.o, ws.attn})
		ws.addNorm(ws.attn, l.mlpNorm)
		ws.project(ws.norm, c.hidden, projection{&l.gate, ws.gate}, projection{&l.up, ws.up})
		ws.run(opSwiGLU, op.rows*swigluChunks(c.intermediate), 4)
		ws.project(ws.gate, c.intermediate, projection{&l.down, ws.attn})
	}
	op.layer = nil
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
		if capP >= gemmAttentionMin {
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
			}
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
	ws.pool.run(&ws.op, items, grain)
}

// addNorm computes h += residual (when residual is non-nil) and
// norm = RMSNorm(h) * weight for every row.
func (ws *Workspace) addNorm(residual, weight []float32) {
	ws.op.residual, ws.op.normWeight = residual, weight
	ws.run(opAddNorm, ws.op.rows, 2)
	ws.op.residual, ws.op.normWeight = nil, nil
}

type projection struct {
	l   *linear
	dst []float32
}

// project multiplies the rows of src (width cols) by up to three projections
// that share that input. Each 16-row tile is scaled and converted once, then
// every panel of every projection runs in one parallel dispatch. Shapes are
// validated when the model loads.
func (ws *Workspace) project(src []float32, cols int, projs ...projection) {
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
	for t := range tiles {
		n := min(q8gemm.ActivationRows, rows-t*q8gemm.ActivationRows)
		var err error
		if op.rot != nil {
			err = ws.tilesI8[t].Prepare(n, cols)
		} else {
			err = ws.tiles[t].Prepare(n, cols)
		}
		if err != nil {
			panic("qwen3: activation tile: " + err.Error())
		}
	}
	op.src, op.cols = src, cols
	if op.rot != nil {
		// Rotate each input row into ws.rotated; the tiles quantize that.
		ws.run(opRotate, rows, 2)
		op.src = ws.rotated
	}
	ws.run(opRowScale, rows, 2)
	ws.run(opPack, tiles*packChunks(cols), 1)
	ws.run(opProject, op.panels, 1)
	op.src, op.rot = nil, nil
	for i := range op.proj {
		op.proj[i] = projection{}
	}
}
