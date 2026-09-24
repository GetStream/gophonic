// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"

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
	q, dst                      []float32
	errors                      []error
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
	headSize := a.state / a.heads
	scale := float32(math.Pow(float64(headSize), -0.25))
	for i := 0; i < a.rows*a.state; i++ {
		q[i] *= scale
		k[i] *= scale
	}
	for head := 0; head < a.heads; head++ {
		offset := head * headSize
		if err := a.keys[head].Pack(k[offset:], a.state, true); err != nil {
			return err
		}
		if err := a.values[head].Pack(v[offset:], a.state, false); err != nil {
			return err
		}
	}
	a.q, a.dst = q, dst
	clear(a.errors)
	err := executor.Rows(a, a.workers, 1)
	a.q, a.dst = nil, nil
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

// ApplyRows implements whispergemm.RowOperation. The row indices select
// private worker scratch; query tiles are interleaved to balance the tail.
func (a *audioAttention) ApplyRows(firstWorker, lastWorker int) {
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
				inverse[r] = softmaxExpRow(scores[r*a.rows : (r+1)*a.rows])
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
