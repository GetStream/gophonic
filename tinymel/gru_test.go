// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package tinymel

import (
	"math"
	"sync"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

func TestTinyGRUAgainstONNXEquationReference(t *testing.T) {
	input, W, R, B := tinyGRUTestTensors()
	got := make([]float32, tinyGRUSequenceLength*tinyGRUOutputSize)
	work := newTinyGRUWorkspace()
	runTinyGRU(input, W, R, B, got, &work)
	want := tinyGRUReference(input, W, R, B)

	maxError := float32(0)
	for i := range got {
		error := float32(math.Abs(float64(got[i] - want[i])))
		if error > maxError {
			maxError = error
		}
	}
	if maxError > 3e-5 {
		t.Fatalf("ONNX GRU output max abs error %.8g exceeds 3e-5", maxError)
	}

	// Reusing the same workspace must not leak the forward terminal state into
	// the reverse direction or retain any state from the prior invocation.
	second := make([]float32, len(got))
	runTinyGRU(input, W, R, B, second, &work)
	for i := range got {
		if got[i] != second[i] {
			t.Fatalf("workspace reuse changed output at %d: %.9g != %.9g", i, got[i], second[i])
		}
	}
}

func TestTinyGRUZeroAllocations(t *testing.T) {
	input, W, R, B := tinyGRUTestTensors()
	output := make([]float32, tinyGRUSequenceLength*tinyGRUOutputSize)
	work := newTinyGRUWorkspace()
	allocs := testing.AllocsPerRun(8, func() {
		runTinyGRU(input, W, R, B, output, &work)
	})
	if allocs != 0 {
		t.Fatalf("runTinyGRU allocated %.2f objects per call", allocs)
	}
	wStride, rStride, bStride := tinyGRUInputStride, tinyGRURecurrentStride, tinyGRUBiasStride
	directionAllocs := testing.AllocsPerRun(8, func() {
		runTinyGRUDirection(
			input,
			W[:wStride], R[:rStride], B[:bStride], output,
			work.directionScratch(0), 0,
		)
	})
	if directionAllocs != 0 {
		t.Fatalf("runTinyGRUDirection allocated %.2f objects per call", directionAllocs)
	}
}

func TestTinyGRUDirectionsConcurrentMatchSerial(t *testing.T) {
	input, W, R, B := tinyGRUTestTensors()
	serial := make([]float32, tinyGRUSequenceLength*tinyGRUOutputSize)
	parallel := make([]float32, len(serial))
	work := newTinyGRUWorkspace()
	runTinyGRU(input, W, R, B, serial, &work)

	var wait sync.WaitGroup
	for direction := 0; direction < tinyGRUDirectionCount; direction++ {
		wStart, rStart, bStart := direction*tinyGRUInputStride, direction*tinyGRURecurrentStride, direction*tinyGRUBiasStride
		w := W[wStart : wStart+tinyGRUInputStride]
		r := R[rStart : rStart+tinyGRURecurrentStride]
		b := B[bStart : bStart+tinyGRUBiasStride]
		wait.Add(1)
		go func(direction int, w, r, b []float32) {
			defer wait.Done()
			runTinyGRUDirection(input, w, r, b, parallel, work.directionScratch(direction), direction)
		}(direction, w, r, b)
	}
	wait.Wait()
	for i := range serial {
		if serial[i] != parallel[i] {
			t.Fatalf("concurrent GRU output[%d] differs from serial: %.9g != %.9g", i, parallel[i], serial[i])
		}
	}
}

func TestTinyGRUSmallONNNSemantics(t *testing.T) {
	dims := tinyGRUDimensions{sequenceLength: 3, inputSize: 2, hiddenSize: 2, directions: 2}
	input := []float32{0.25, -0.5, 0.75, 0.125, -0.25, 0.625}
	inputStride := tinyGRUGateCount * dims.hiddenSize * dims.inputSize
	recurrentStride := tinyGRUGateCount * dims.hiddenSize * dims.hiddenSize
	biasStride := 2 * tinyGRUGateCount * dims.hiddenSize
	W := make([]float32, dims.directions*inputStride)
	R := make([]float32, dims.directions*recurrentStride)
	B := make([]float32, dims.directions*biasStride)
	// Use distinct directional weights and nonzero Wb/Rb for every gate. This
	// exercises direction indexing, reverse-time traversal, z/r/h gate order,
	// and the linear_before_reset=1 candidate equation.
	for direction := 0; direction < dims.directions; direction++ {
		w := W[direction*inputStride : (direction+1)*inputStride]
		r := R[direction*recurrentStride : (direction+1)*recurrentStride]
		b := B[direction*biasStride : (direction+1)*biasStride]
		sign := float32(1)
		if direction == 1 {
			sign = -1
		}
		for i := range w {
			w[i] = sign * float32((i%7)-3) * 0.09
		}
		for i := range r {
			r[i] = sign * float32((i%5)-2) * 0.07
		}
		for i := range b {
			b[i] = float32((i%9)-4) * 0.035
		}
	}
	output := make([]float32, dims.sequenceLength*dims.directions*dims.hiddenSize)
	inputGates := make([]float32, dims.directions*dims.sequenceLength*tinyGRUGateCount*dims.hiddenSize)
	runTinyGRUWithDimensions(input, W, R, B, output, inputGates, dims)
	want := tinyGRUReferenceWithDimensions(input, W, R, B, dims)
	for i := range output {
		if delta := math.Abs(float64(output[i] - want[i])); delta > 3e-6 {
			t.Fatalf("small ONNX GRU output[%d]: got %.9g, want %.9g (delta %.3g)", i, output[i], want[i], delta)
		}
	}
}

func TestTinyGRUOnnxRuntimeStageFixture(t *testing.T) {
	path := testmodels.Path(t, testmodels.TinyMel)
	model, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	input := readFloatFixture(t, "testdata/tone_gru_input.f32le")
	want := readFloatFixture(t, "testdata/tone_gru_output.f32le")
	wantFinal := readFloatFixture(t, "testdata/tone_gru_final_hidden.f32le")
	work := newTinyGRUWorkspace()
	var wait sync.WaitGroup
	for direction := 0; direction < tinyGRUDirectionCount; direction++ {
		wStart, rStart, bStart := direction*tinyGRUInputStride, direction*tinyGRURecurrentStride, direction*tinyGRUBiasStride
		w := model.gruW[wStart : wStart+tinyGRUInputStride]
		r := model.gruR[rStart : rStart+tinyGRURecurrentStride]
		b := model.gruB[bStart : bStart+tinyGRUBiasStride]
		wait.Add(1)
		go func(direction int, w, r, b []float32) {
			defer wait.Done()
			runTinyGRUDirection(input, w, r, b, work.buffer(), work.directionScratch(direction), direction)
		}(direction, w, r, b)
	}
	wait.Wait()

	maxError := float32(0)
	for i, expected := range want {
		if delta := float32(math.Abs(float64(work.buffer()[i] - expected))); delta > maxError {
			maxError = delta
		}
	}
	if maxError > 2e-5 {
		t.Fatalf("ORT GRU output max abs error %.8g exceeds 2e-5", maxError)
	}
	for direction := 0; direction < tinyGRUDirectionCount; direction++ {
		lastTime := tinyGRUSequenceLength - 1
		if direction == 1 {
			lastTime = 0
		}
		for hidden := 0; hidden < tinyGRUHiddenSize; hidden++ {
			got := work.buffer()[lastTime*tinyGRUOutputSize+direction*tinyGRUHiddenSize+hidden]
			want := wantFinal[direction*tinyGRUHiddenSize+hidden]
			if delta := math.Abs(float64(got - want)); delta > 2e-5 {
				t.Fatalf("ORT final hidden [%d,%d]: got %.9g, want %.9g (delta %.3g)", direction, hidden, got, want, delta)
			}
		}
	}
}

func BenchmarkTinyGRU(b *testing.B) {
	input, W, R, B := tinyGRUTestTensors()
	output := make([]float32, tinyGRUSequenceLength*tinyGRUOutputSize)
	work := newTinyGRUWorkspace()
	b.ReportAllocs()
	b.SetBytes(int64(len(input)*4 + len(W)*4 + len(R)*4 + len(B)*4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runTinyGRU(input, W, R, B, output, &work)
	}
}

func tinyGRUTestTensors() (input, W, R, B []float32) {
	input = make([]float32, tinyGRUSequenceLength*tinyGRUInputSize)
	W = make([]float32, tinyGRUDirectionCount*tinyGRUInputStride)
	R = make([]float32, tinyGRUDirectionCount*tinyGRURecurrentStride)
	B = make([]float32, tinyGRUDirectionCount*tinyGRUBiasStride)
	seed := uint32(0x9e3779b9)
	next := func(scale float32) float32 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return (float32(seed>>8)/float32(1<<24)*2 - 1) * scale
	}
	for i := range input {
		input[i] = next(0.5)
	}
	for i := range W {
		W[i] = next(0.015)
	}
	for i := range R {
		R[i] = next(0.012)
	}
	for i := range B {
		B[i] = next(0.005)
	}
	return input, W, R, B
}

// tinyGRUReference is a direct scalar transcription of the ONNX equations.
// It deliberately recomputes XW in the timestep loop and uses the standard
// library activations, independently checking the optimized kernel's cached
// projections and fast float32 activations.
func tinyGRUReference(input, W, R, B []float32) []float32 {
	return tinyGRUReferenceWithDimensions(input, W, R, B, tinyMelGRUDimensions)
}

func tinyGRUReferenceWithDimensions(input, W, R, B []float32, dims tinyGRUDimensions) []float32 {
	sequenceLength, inputSize, hiddenSize, directions := dims.sequenceLength, dims.inputSize, dims.hiddenSize, dims.directions
	gateSize := tinyGRUGateCount * hiddenSize
	inputStride := gateSize * inputSize
	recurrentStride := gateSize * hiddenSize
	biasStride := 2 * gateSize
	outputSize := directions * hiddenSize
	output := make([]float32, sequenceLength*outputSize)
	for direction := 0; direction < directions; direction++ {
		w := W[direction*inputStride : (direction+1)*inputStride]
		r := R[direction*recurrentStride : (direction+1)*recurrentStride]
		bias := B[direction*biasStride : (direction+1)*biasStride]
		var previousStorage [tinyGRUHiddenSize]float32
		previous := previousStorage[:hiddenSize]
		first, end, step := 0, sequenceLength, 1
		if direction == 1 {
			first, end, step = sequenceLength-1, -1, -1
		}
		for t := first; t != end; t += step {
			x := input[t*inputSize : (t+1)*inputSize]
			var nextStorage [tinyGRUHiddenSize]float32
			next := nextStorage[:hiddenSize]
			for unit := 0; unit < hiddenSize; unit++ {
				z := tinyGRUSigmoidReference(
					tinyGRUDot(w[unit*inputSize:(unit+1)*inputSize], x) +
						tinyGRUDot(r[unit*hiddenSize:(unit+1)*hiddenSize], previous) + bias[unit] + bias[gateSize+unit],
				)
				rGate := tinyGRUSigmoidReference(
					tinyGRUDot(w[hiddenSize*inputSize+unit*inputSize:hiddenSize*inputSize+(unit+1)*inputSize], x) +
						tinyGRUDot(r[hiddenSize*hiddenSize+unit*hiddenSize:hiddenSize*hiddenSize+(unit+1)*hiddenSize], previous) + bias[hiddenSize+unit] + bias[gateSize+hiddenSize+unit],
				)
				wHiddenOffset := 2 * hiddenSize * inputSize
				rHiddenOffset := 2 * hiddenSize * hiddenSize
				xh := tinyGRUDot(w[wHiddenOffset+unit*inputSize:wHiddenOffset+(unit+1)*inputSize], x)
				rh := tinyGRUDot(r[rHiddenOffset+unit*hiddenSize:rHiddenOffset+(unit+1)*hiddenSize], previous)
				candidate := float32(math.Tanh(float64(xh + bias[2*hiddenSize+unit] + rGate*(rh+bias[gateSize+2*hiddenSize+unit]))))
				z = (1-z)*candidate + z*previous[unit]
				next[unit] = z
			}
			copy(previous, next)
			row := output[t*outputSize+direction*hiddenSize : t*outputSize+(direction+1)*hiddenSize]
			copy(row, previous)
		}
	}
	return output
}

func tinyGRUDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func tinyGRUSigmoidReference(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}
