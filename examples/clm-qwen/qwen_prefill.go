// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"errors"
	"fmt"
	"math"

	"github.com/GetStream/gophonic/internal/q8gemm"
)

// Evaluator computes Qwen3 last-token hidden states a layer at a time. All
// tokens of a call, including tokens from several independent sequences, pass
// through each projection together, so each packed weight tile is read once
// per 16 tokens instead of once per token or per sequence.
type Evaluator struct {
	m *Model
}

// NewEvaluator returns an evaluator over m. It holds no mutable state; create
// one Workspace per concurrent call.
func NewEvaluator(m *Model) (*Evaluator, error) {
	if m == nil {
		return nil, errors.New("clmqwen: nil Qwen3 model")
	}
	return &Evaluator{m: m}, nil
}

// Workspace owns reusable activations for a packed batch of rows, one layer's
// keys and values, per-tile activation scratch, and optionally a pool of
// persistent workers. It belongs to one evaluator and one concurrent call.
type Workspace struct {
	h, norm, q, ctx, attn, gate, up []float32
	keys, values                    []float32 // K and V projections of the current layer, [rows][kvDim]
	rowStart, rowPos                []int32   // first row of each row's sequence, and its position
	ropeCos, ropeSin                []float32
	ropePositions                   int
	capacity                        int
	owner                           *Evaluator
	tiles                           []*q8gemm.Workspace
	scratch                         []*q8gemm.Scratch // per participant; nil entries with SME
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
		return nil, errors.New("clmqwen: nil evaluator")
	}
	if workers < 1 || workers > 64 {
		return nil, fmt.Errorf("clmqwen: invalid worker count %d", workers)
	}
	ws := &Workspace{owner: e}
	ws.op.ws = ws
	if workers > 1 {
		ws.pool = newWorkerPool(workers)
	}
	c := &e.m.cfg
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
		return errors.New("clmqwen: nil workspace")
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
		return errors.New("clmqwen: nil evaluator or workspace")
	}
	if ws.owner != e {
		return errors.New("clmqwen: workspace belongs to another evaluator")
	}
	m, c := e.m, &e.m.cfg
	if len(seqs) == 0 || len(dst) != len(seqs) {
		return fmt.Errorf("clmqwen: %d destinations for %d sequences", len(dst), len(seqs))
	}
	rows, longest := 0, 0
	for s, ids := range seqs {
		if len(ids) == 0 {
			return errors.New("clmqwen: empty token sequence")
		}
		if len(ids) > c.maxPositions {
			return fmt.Errorf("clmqwen: %d tokens exceeds Qwen3 context %d", len(ids), c.maxPositions)
		}
		if len(dst[s]) != c.hidden {
			return fmt.Errorf("clmqwen: destination width %d, want %d", len(dst[s]), c.hidden)
		}
		for i, id := range ids {
			if id < 0 || id >= c.vocab {
				return fmt.Errorf("clmqwen: token %d at position %d is outside vocabulary", id, i)
			}
		}
		if rows > int(^uint(0)>>2)-len(ids) {
			return errors.New("clmqwen: batch size overflows int")
		}
		rows += len(ids)
		longest = max(longest, len(ids))
	}
	if err := ws.ensure(c, rows, longest); err != nil {
		return err
	}
	row := 0
	for _, ids := range seqs {
		start := int32(row)
		for pos, id := range ids {
			m.embedRow(id, ws.h[row*c.hidden:(row+1)*c.hidden])
			ws.rowStart[row], ws.rowPos[row] = start, int32(pos)
			row++
		}
	}
	ws.prepareRoPE(c, longest)

	op := &ws.op
	op.rows = rows
	for layer := range m.layers {
		l := &m.layers[layer]
		op.layer = l
		// The previous layer's MLP output waits in attn; fold that residual
		// add into this layer's attention RMSNorm.
		residual := ws.attn
		if layer == 0 {
			residual = nil
		}
		ws.addNorm(residual, l.attnNorm)
		ws.project(ws.norm, c.hidden, projection{l.q, ws.q}, projection{l.k, ws.keys}, projection{l.v, ws.values})
		ws.run(opQKRope, rows, 4)
		ws.run(opAttention, rows*c.kvHeads, 4)
		ws.project(ws.ctx, c.heads*c.headDim, projection{l.o, ws.attn})
		ws.addNorm(ws.attn, l.mlpNorm)
		ws.project(ws.norm, c.hidden, projection{l.gate, ws.gate}, projection{l.up, ws.up})
		ws.run(opSwiGLU, rows*swigluChunks(c.intermediate), 4)
		ws.project(ws.gate, c.intermediate, projection{l.down, ws.attn})
	}
	op.layer = nil
	row = 0
	for s, ids := range seqs {
		row += len(ids)
		last := ws.h[(row-1)*c.hidden : row*c.hidden]
		addInto(last, ws.attn[(row-1)*c.hidden:row*c.hidden])
		rmsNorm32(dst[s], last, m.finalNorm, c.eps)
		for _, value := range dst[s] {
			if !finite32(value) {
				return errors.New("clmqwen: non-finite hidden state")
			}
		}
	}
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
				return errors.New("clmqwen: workspace size overflows int")
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
		tiles := (capN + q8gemm.ActivationRows - 1) / q8gemm.ActivationRows
		for len(ws.tiles) < tiles {
			tile, err := q8gemm.NewWorkspace(max(c.hidden, qdim, c.intermediate))
			if err != nil {
				return fmt.Errorf("clmqwen: activation tile: %w", err)
			}
			ws.tiles = append(ws.tiles, tile)
		}
		ws.capacity = capN
	}
	if positions > len(ws.ropeCos)/(c.headDim/2) {
		capP := max(16, positions)
		ws.ropeCos = make([]float32, capP*c.headDim/2)
		ws.ropeSin = make([]float32, capP*c.headDim/2)
		ws.ropePositions = 0
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
	w   *q8gemm.Weights
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
		op.panelEnd[i] = op.panels + p.w.Panels()
		op.panels = op.panelEnd[i]
	}
	tiles := (rows + q8gemm.ActivationRows - 1) / q8gemm.ActivationRows
	for t := range tiles {
		if err := ws.tiles[t].Prepare(min(q8gemm.ActivationRows, rows-t*q8gemm.ActivationRows), cols); err != nil {
			panic("clmqwen: activation tile: " + err.Error())
		}
	}
	op.src, op.cols = src, cols
	ws.run(opRowScale, rows, 2)
	ws.run(opPack, tiles*packChunks(cols), 1)
	ws.run(opProject, op.panels, 1)
	op.src = nil
	for i := range op.proj {
		op.proj[i] = projection{}
	}
}
