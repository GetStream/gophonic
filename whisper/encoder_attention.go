// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"

	"github.com/GetStream/gophonic/internal/nn"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

const attentionTileRows = 32

// audioAttention assigns independent query tiles to the persistent workers.
// Each tile computes QK, stable softmax, and the value product before moving on,
// keeping its scores in private scratch and requiring one worker handoff for
// the complete attention operation. The packed K/V matrices are read-only
// during this phase; no complete time-by-time score matrix is materialized.
type audioAttention struct {
	rows, state, heads, workers int
	keys, values                []*whispergemm.PackedB
	scores                      []float32
	q, k, v, dst                []float32
	queryBias, valueBias        []float32
	errors                      []error
	preparing, rowPrep, packing bool
}

func newAudioAttention(rows, state, heads, workers int) (*audioAttention, error) {
	a := &audioAttention{
		rows: rows, state: state, heads: heads, workers: workers,
		keys: make([]*whispergemm.PackedB, heads), values: make([]*whispergemm.PackedB, heads),
		scores: make([]float32, workers*attentionTileRows*rows), errors: make([]error, workers),
	}
	for head := 0; head < heads; head++ {
		var err error
		a.keys[head], err = whispergemm.NewPackedB(state/heads, rows)
		if err != nil {
			return nil, err
		}
		a.values[head], err = whispergemm.NewPackedB(rows, state/heads)
		if err != nil {
			return nil, err
		}
	}
	return a, nil
}

// run permits dst to alias q: each query tile consumes its Q values before
// overwriting them. K and V must remain disjoint from Q and the destination.
func (a *audioAttention) run(q, k, v, dst []float32, executor *whispergemm.Executor) error {
	return a.runBiased(q, k, v, dst, nil, nil, executor)
}

// runBiased first adds the optional Q and V projection biases, scales Q and K,
// and packs each head's K and V, all in parallel over heads. Each element is
// computed as (q+bias)*scale, k*scale, and v+bias, as separate passes would.
func (a *audioAttention) runBiased(q, k, v, dst, queryBias, valueBias []float32, executor *whispergemm.Executor) error {
	a.q, a.k, a.v, a.dst = q, k, v, dst
	a.queryBias, a.valueBias = queryBias, valueBias
	clear(a.errors)
	// Whole rows first: one NEON pass per row when both biases are present.
	a.preparing = true
	a.rowPrep = nn.Accelerated && queryBias != nil && valueBias != nil && a.state%4 == 0
	err := executor.Rows(a, a.heads, 1)
	if err == nil && a.rowPrep {
		a.packing = true
		err = executor.Rows(a, a.heads, 1)
		a.packing = false
	}
	a.preparing = false
	if err == nil {
		err = executor.Rows(a, a.workers, 1)
	}
	a.q, a.k, a.v, a.dst, a.queryBias, a.valueBias = nil, nil, nil, nil, nil, nil
	if err != nil {
		return err
	}
	for _, err := range a.errors {
		if err != nil {
			return err
		}
	}
	return nil
}

func (a *audioAttention) prepareHeads(firstHead, lastHead int) {
	headSize := a.state / a.heads
	scale := float32(math.Pow(float64(headSize), -0.25))
	for head := firstHead; head < lastHead; head++ {
		offset := head * headSize
		if a.rowPrep {
			// This head's share of rows gets the full-row elementwise pass;
			// every head's columns are ready before any head packs below.
			for t := head * a.rows / a.heads; t < (head+1)*a.rows/a.heads; t++ {
				base := t * a.state
				attnPrepNEON(&a.q[base], &a.k[base], &a.v[base], &a.queryBias[0], &a.valueBias[0], a.state, scale)
			}
			continue
		}
		for t := 0; t < a.rows; t++ {
			base := t*a.state + offset
			q, k, v := a.q[base:base+headSize], a.k[base:base+headSize], a.v[base:base+headSize]
			if a.queryBias != nil {
				qb := a.queryBias[offset : offset+headSize]
				for i := range q {
					q[i] = (q[i] + qb[i]) * scale
				}
			} else {
				for i := range q {
					q[i] *= scale
				}
			}
			for i := range k {
				k[i] *= scale
			}
			if a.valueBias != nil {
				vb := a.valueBias[offset : offset+headSize]
				for i := range v {
					v[i] += vb[i]
				}
			}
		}
		a.packHead(head)
	}
}

func (a *audioAttention) packHead(head int) {
	offset := head * (a.state / a.heads)
	if err := a.keys[head].Pack(a.k[offset:], a.state, true); err != nil {
		a.errors[head%a.workers] = err
		return
	}
	if err := a.values[head].Pack(a.v[offset:], a.state, false); err != nil {
		a.errors[head%a.workers] = err
	}
}

// ApplyRows implements whispergemm.RowOperation. The row indices select
// private worker scratch; query tiles are interleaved to balance the tail.
func (a *audioAttention) ApplyRows(firstWorker, lastWorker int) {
	if a.packing {
		for head := firstWorker; head < lastWorker; head++ {
			a.packHead(head)
		}
		return
	}
	if a.preparing {
		a.prepareHeads(firstWorker, lastWorker)
		return
	}
	tilesPerHead := (a.rows + attentionTileRows - 1) / attentionTileRows
	headSize := a.state / a.heads
	for worker := firstWorker; worker < lastWorker; worker++ {
		scores := a.scores[worker*attentionTileRows*a.rows : (worker+1)*attentionTileRows*a.rows]
		for tile := worker; tile < a.heads*tilesPerHead; tile += a.workers {
			head := tile / tilesPerHead
			firstRow := (tile % tilesPerHead) * attentionTileRows
			rows := min(attentionTileRows, a.rows-firstRow)
			offset := firstRow*a.state + head*(a.state/a.heads)
			if err := a.keys[head].Mul(scores, a.rows, a.q[offset:], a.state, rows); err != nil {
				a.errors[worker] = err
				return
			}
			var inverse [attentionTileRows]float32
			for r := 0; r < rows; r++ {
				inverse[r] = nn.SoftmaxExp(scores[r*a.rows : (r+1)*a.rows])
			}
			if err := a.values[head].Mul(a.dst[offset:], a.state, scores, a.rows, rows); err != nil {
				a.errors[worker] = err
				return
			}
			// Normalize the headSize-wide product instead of every score.
			for r := 0; r < rows; r++ {
				out := a.dst[offset+r*a.state : offset+r*a.state+headSize]
				for i := range out {
					out[i] *= inverse[r]
				}
			}
		}
	}
}

func softmaxRows(values []float32, rows, columns int) {
	row := 0
	for ; row+4 <= rows; row += 4 {
		softmaxFourRows(
			values[row*columns:(row+1)*columns], values[(row+1)*columns:(row+2)*columns],
			values[(row+2)*columns:(row+3)*columns], values[(row+3)*columns:(row+4)*columns],
		)
	}
	for ; row < rows; row++ {
		softmaxRow(values[row*columns : (row+1)*columns])
	}
}

func softmaxRow(x []float32) {
	maxValue := x[0]
	for _, value := range x[1:] {
		maxValue = max(maxValue, value)
	}
	total := softmaxExpInPlace(x, maxValue)
	inverse := 1 / total
	for i := range x {
		x[i] *= inverse
	}
}
