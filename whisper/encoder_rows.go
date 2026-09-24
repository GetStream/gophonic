// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

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
			layerNormRow(op.dst[r*w:(r+1)*w], op.out[r*w:(r+1)*w], op.normW, op.normB)
		}
	case rowsResidual:
		if layerNormAccelerated && op.normW != nil && w > 0 && w%8 == 0 && &op.src[0] == &op.out[0] {
			// Fused: row += add + bias, then out = LayerNorm(row). out may
			// alias add; each chunk of add is read before out is written.
			var bias *float32
			if op.bias != nil {
				bias = &op.bias[0]
			}
			for r := start; r < end; r++ {
				residualNormNEON(&op.dst[r*w], &op.out[r*w], &op.normW[0], &op.normB[0], w, &op.src[r*w], bias)
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
				layerNormRow(row, op.out[r*w:(r+1)*w], op.normW, op.normB)
			}
		}
	case rowsBias:
		addRowBias(op.dst[start*w:end*w], op.bias, end-start, w)
	case rowsPosition:
		addPositionEmbedding(op.dst[start*w:end*w], op.src[start*w:end*w])
	case rowsLowerChannel:
		lowerChannelMajor3Rows(op.src, op.dst, op.frames, w, start, end)
	case rowsLowerTime:
		lowerTimeMajor3Stride2Rows(op.src, op.dst, op.frames, op.outRows, w, start, end)
	}
}

func (w *EncoderWorkspace) rows(op encoderRows, rows int) error {
	w.rowOp = op
	err := w.gemm.Rows(&w.rowOp, rows, 16)
	w.rowOp = encoderRows{}
	return err
}
