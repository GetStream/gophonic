// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import "github.com/GetStream/gophonic/internal/nn"

// encoderRowKind selects the elementwise work of one parallel encoder pass.
type encoderRowKind uint8

const (
	// rowsNorm writes out[r] = LayerNorm(dst[r]).
	rowsNorm encoderRowKind = iota
	// rowsResidual adds dst[r] += src[r] + bias, then optionally writes
	// out[r] = LayerNorm(dst[r]) when norm weights are present.
	rowsResidual
	// rowsBias adds bias to every row of dst.
	rowsBias
	// rowsPosition adds the fixed positional embedding to dst.
	rowsPosition
	// rowsLowerChannel and rowsLowerTime build the stem convolution columns.
	rowsLowerChannel
	rowsLowerTime
	// rowsTransposeMel writes time-major rows from the channel-major mel.
	rowsTransposeMel
)

// encoderRows is stored in the workspace so dispatch does not allocate. Every
// value is produced by the same operations as the former serial loops, so
// results do not depend on the worker count.
type encoderRows struct {
	kind                   encoderRowKind
	dst, src, out, bias    []float32
	normW, normB           []float32
	width, frames, outRows int
}

func (op *encoderRows) ApplyRows(start, end int) {
	w := op.width
	switch op.kind {
	case rowsNorm:
		for r := start; r < end; r++ {
			nn.LayerNorm(op.dst[r*w:(r+1)*w], op.out[r*w:(r+1)*w], op.normW, op.normB)
		}
	case rowsResidual:
		if nn.Accelerated && op.normW != nil && w > 0 && w%8 == 0 && &op.src[0] == &op.out[0] {
			// Fused: row += add + bias, then out = LayerNorm(row). out may
			// alias add; each chunk of add is read before out is written.
			for r := start; r < end; r++ {
				nn.ResidualNorm(op.dst[r*w:(r+1)*w], op.out[r*w:(r+1)*w], op.normW, op.normB, op.src[r*w:(r+1)*w], op.bias)
			}
			return
		}
		for r := start; r < end; r++ {
			row, add := op.dst[r*w:(r+1)*w], op.src[r*w:(r+1)*w]
			if op.bias != nil {
				bias := op.bias[:w]
				for i := range row {
					row[i] += add[i] + bias[i]
				}
			} else {
				for i := range row {
					row[i] += add[i]
				}
			}
			if op.normW != nil {
				nn.LayerNorm(row, op.out[r*w:(r+1)*w], op.normW, op.normB)
			}
		}
	case rowsBias:
		nn.AddRowBias(op.dst[start*w:end*w], op.bias, end-start, w)
	case rowsPosition:
		addPositionEmbedding(op.dst[start*w:end*w], op.src[start*w:end*w])
	case rowsLowerChannel:
		lowerChannelMajor3Rows(op.src, op.dst, op.frames, w, start, end)
	case rowsTransposeMel:
		for t := start; t < end; t++ {
			row := op.dst[t*w : (t+1)*w]
			for c := range row {
				row[c] = op.src[c*op.frames+t]
			}
		}
	case rowsLowerTime:
		lowerTimeMajor3Stride2Rows(op.src, op.dst, op.frames, op.outRows, w, start, end)
	}
}

func (w *encoderWorkspace) rows(op encoderRows, rows int) error {
	w.rowOp = op
	err := w.gemm.Rows(&w.rowOp, rows, 16)
	w.rowOp = encoderRows{}
	return err
}
