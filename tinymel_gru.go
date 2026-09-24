// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

const (
	tinyGRUSequenceLength  = 100
	tinyGRUInputSize       = 192
	tinyGRUHiddenSize      = 128
	tinyGRUDirectionCount  = 2
	tinyGRUGateCount       = 3
	tinyGRUGateSize        = tinyGRUGateCount * tinyGRUHiddenSize
	tinyGRUOutputSize      = tinyGRUDirectionCount * tinyGRUHiddenSize
	tinyGRUInputStride     = tinyGRUGateSize * tinyGRUInputSize
	tinyGRURecurrentStride = tinyGRUGateSize * tinyGRUHiddenSize
	tinyGRUBiasStride      = 2 * tinyGRUGateSize
)

type tinyGRUDimensions struct {
	sequenceLength int
	inputSize      int
	hiddenSize     int
	directions     int
}

var tinyMelGRUDimensions = tinyGRUDimensions{
	sequenceLength: tinyGRUSequenceLength,
	inputSize:      tinyGRUInputSize,
	hiddenSize:     tinyGRUHiddenSize,
	directions:     tinyGRUDirectionCount,
}

// tinyGRUWorkspace owns the input projections reused while evaluating each
// direction. A workspace is private to one prediction and is reusable.
type tinyGRUWorkspace struct {
	inputGates []float32 // [direction, time, z/r/h * hidden]
	output     []float32 // [time, forward hidden || backward hidden]
}

func newTinyGRUWorkspace() tinyGRUWorkspace {
	return tinyGRUWorkspace{
		inputGates: make([]float32, tinyGRUDirectionCount*tinyGRUSequenceLength*tinyGRUGateSize),
		output:     make([]float32, tinyGRUSequenceLength*tinyGRUOutputSize),
	}
}

func (work *tinyGRUWorkspace) buffer() []float32 { return work.output }

// directionScratch returns the projection buffer owned by one direction. The
// returned slices for forward and reverse never overlap.
func (work *tinyGRUWorkspace) directionScratch(direction int) []float32 {
	if direction < 0 || direction >= tinyGRUDirectionCount {
		panic("gophonic: invalid TinyMelNet GRU direction")
	}
	stride := tinyGRUSequenceLength * tinyGRUGateSize
	return work.inputGates[direction*stride : (direction+1)*stride]
}

// runTinyGRU evaluates the fixed bidirectional TinyMelNet GRU. input is
// time-major [100,192], W and R use ONNX [direction, z/r/h * hidden, input]
// row-major storage, B stores Wb[z/r/h] followed by Rb[z/r/h] per direction,
// and output is time-major [100, forward-hidden || backward-hidden].
//
// The function follows the ONNX GRU equations with linear_before_reset=1:
// z/r use XW + HR + Wb + Rb; the candidate uses XWh + Wbh + r*(HRh + Rbh).
// The reverse direction traverses input backwards but writes each state at
// its original sequence position, as required by ONNX Y.
func runTinyGRU(input, W, R, B, output []float32, work *tinyGRUWorkspace) {
	if len(input) != tinyGRUSequenceLength*tinyGRUInputSize ||
		len(W) != tinyGRUDirectionCount*tinyGRUInputStride ||
		len(R) != tinyGRUDirectionCount*tinyGRURecurrentStride ||
		len(B) != tinyGRUDirectionCount*tinyGRUBiasStride ||
		len(output) != tinyGRUSequenceLength*tinyGRUOutputSize ||
		work == nil || len(work.inputGates) != tinyGRUDirectionCount*tinyGRUSequenceLength*tinyGRUGateSize ||
		len(work.output) != tinyGRUSequenceLength*tinyGRUOutputSize {
		panic("gophonic: invalid TinyMelNet GRU tensor or workspace size")
	}
	runTinyGRUWithDimensions(input, W, R, B, output, work.inputGates, tinyMelGRUDimensions)
}

// runTinyGRUWithDimensions is the small, allocation-free kernel core. Its
// scratch slices are supplied by the caller so small synthetic dimensions can
// exercise the same ONNX storage and recurrence logic in tests.
func runTinyGRUWithDimensions(input, W, R, B, output, inputGates []float32, dims tinyGRUDimensions) {
	sequenceLength, inputSize, hiddenSize, directions := dims.sequenceLength, dims.inputSize, dims.hiddenSize, dims.directions
	gateSize := tinyGRUGateCount * hiddenSize
	inputStride := gateSize * inputSize
	recurrentStride := gateSize * hiddenSize
	biasStride := 2 * gateSize
	outputSize := directions * hiddenSize
	if sequenceLength <= 0 || inputSize <= 0 || hiddenSize <= 0 || hiddenSize > tinyGRUHiddenSize || directions != tinyGRUDirectionCount ||
		len(input) != sequenceLength*inputSize || len(W) != directions*inputStride ||
		len(R) != directions*recurrentStride || len(B) != directions*biasStride ||
		len(output) != sequenceLength*outputSize ||
		len(inputGates) != directions*sequenceLength*gateSize {
		panic("gophonic: invalid TinyMelNet GRU dimensions or tensor size")
	}
	for direction := 0; direction < directions; direction++ {
		w := W[direction*inputStride : (direction+1)*inputStride]
		r := R[direction*recurrentStride : (direction+1)*recurrentStride]
		bias := B[direction*biasStride : (direction+1)*biasStride]
		scratch := inputGates[direction*sequenceLength*gateSize : (direction+1)*sequenceLength*gateSize]
		runTinyGRUDirectionWithDimensions(input, w, r, bias, output, scratch, direction, dims)
	}
}

// runTinyGRUDirection evaluates one ONNX direction. Wdir, Rdir, and Bdir are
// direction-local ONNX tensors. output is shared between directions, but each
// call writes only its own hidden-size region in every timestep row. inputGates
// is direction-private [time, z/r/h * hidden] scratch.
func runTinyGRUDirection(input, Wdir, Rdir, Bdir, output, inputGates []float32, direction int) {
	if len(Wdir) != tinyGRUInputStride || len(Rdir) != tinyGRURecurrentStride ||
		len(Bdir) != tinyGRUBiasStride || len(inputGates) != tinyGRUSequenceLength*tinyGRUGateSize ||
		direction < 0 || direction >= tinyGRUDirectionCount || len(output) != tinyGRUSequenceLength*tinyGRUOutputSize {
		panic("gophonic: invalid TinyMelNet GRU direction tensor or workspace size")
	}
	runTinyGRUDirectionWithDimensions(input, Wdir, Rdir, Bdir, output, inputGates, direction, tinyMelGRUDimensions)
}

func runTinyGRUDirectionWithDimensions(input, W, R, B, output, inputGates []float32, direction int, dims tinyGRUDimensions) {
	sequenceLength, inputSize, hiddenSize, directions := dims.sequenceLength, dims.inputSize, dims.hiddenSize, dims.directions
	gateSize := tinyGRUGateCount * hiddenSize
	inputStride := gateSize * inputSize
	recurrentStride := gateSize * hiddenSize
	biasStride := 2 * gateSize
	outputSize := directions * hiddenSize
	if sequenceLength <= 0 || inputSize <= 0 || hiddenSize <= 0 || hiddenSize > tinyGRUHiddenSize ||
		directions != tinyGRUDirectionCount || direction < 0 || direction >= directions ||
		len(input) != sequenceLength*inputSize || len(W) != inputStride || len(R) != recurrentStride ||
		len(B) != biasStride || len(output) != sequenceLength*outputSize ||
		len(inputGates) != sequenceLength*gateSize {
		panic("gophonic: invalid TinyMelNet GRU direction dimensions or tensor size")
	}

	// XW does not depend on recurrent state. Project all three gates once for
	// this direction so the timestep loop only performs recurrent GEMVs and
	// gate arithmetic. The ARM64 SIMD kernel batches adjacent timesteps while
	// preserving the original dot-product reduction order.
	tinyGRUProjectInputs(input, W, inputGates, sequenceLength, inputSize, gateSize)

	var stateStorage, nextStorage [tinyGRUHiddenSize]float32
	var recurrentStorage [tinyGRUGateSize]float32
	state, next := stateStorage[:hiddenSize], nextStorage[:hiddenSize]
	recurrent := recurrentStorage[:gateSize]
	first, end, step := 0, sequenceLength, 1
	if direction == 1 {
		first, end, step = sequenceLength-1, -1, -1
	}
	for t := first; t != end; t += step {
		for gate := 0; gate < tinyGRUGateCount; gate++ {
			gateWeights := R[gate*hiddenSize*hiddenSize : (gate+1)*hiddenSize*hiddenSize]
			tinyGRUMatVec(state, gateWeights, recurrent[gate*hiddenSize:(gate+1)*hiddenSize], hiddenSize, hiddenSize)
		}

		projected := inputGates[t*gateSize : (t+1)*gateSize]
		for unit := 0; unit < hiddenSize; unit++ {
			z := tinyGRUSigmoid(projected[unit] + recurrent[unit] + B[unit] + B[gateSize+unit])
			r := tinyGRUSigmoid(projected[hiddenSize+unit] + recurrent[hiddenSize+unit] + B[hiddenSize+unit] + B[gateSize+hiddenSize+unit])
			candidate := tinyGRUTanh(projected[2*hiddenSize+unit] + B[2*hiddenSize+unit] + r*(recurrent[2*hiddenSize+unit]+B[gateSize+2*hiddenSize+unit]))
			next[unit] = (1-z)*candidate + z*state[unit]
		}
		state, next = next, state
		row := output[t*outputSize+direction*hiddenSize : t*outputSize+(direction+1)*hiddenSize]
		copy(row, state)
	}
}

// tinyGRUMatVec computes a row-major matrix times one vector four output rows
// at a time, reusing each input vector load across those rows through dotProduct4.
func tinyGRUMatVec(input, weights, output []float32, inputSize, outputSize int) {
	unit := 0
	for ; unit+4 <= outputSize; unit += 4 {
		row0 := weights[unit*inputSize : (unit+1)*inputSize]
		row1 := weights[(unit+1)*inputSize : (unit+2)*inputSize]
		row2 := weights[(unit+2)*inputSize : (unit+3)*inputSize]
		row3 := weights[(unit+3)*inputSize : (unit+4)*inputSize]
		y0, y1, y2, y3 := dotProduct4(input, row0, row1, row2, row3)
		output[unit], output[unit+1], output[unit+2], output[unit+3] = y0, y1, y2, y3
	}
	for ; unit < outputSize; unit++ {
		row := weights[unit*inputSize : (unit+1)*inputSize]
		output[unit] = dotProduct(row, input)
	}
}

func tinyGRUSigmoid(x float32) float32 {
	if x >= 0 {
		return 1 / (1 + expNegative32(-x))
	}
	e := expNegative32(x)
	return e / (1 + e)
}

func tinyGRUTanh(x float32) float32 {
	return tanhApprox32(x)
}
