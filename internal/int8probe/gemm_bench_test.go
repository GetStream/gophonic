// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package int8probe

import "testing"

const (
	probeActivationScale = float32(0.15650022)
	probeActivationZero  = int32(104)
)

var probeResult float32

type probeData struct {
	aQ       []uint8
	wQ       []int8
	wPacked  []int8
	wSum     []int32
	aScale   float32
	wScale   []float32
	bias     []float32
	aFloat   []float32
	wFloat   []float32
	outFloat []float32
	outInt8  []float32
}

func makeProbeData() *probeData {
	d := &probeData{
		aQ:       make([]uint8, benchRows*benchK),
		wQ:       make([]int8, benchCols*benchK),
		wPacked:  make([]int8, benchK*benchCols),
		wSum:     make([]int32, benchCols),
		aScale:   probeActivationScale,
		wScale:   make([]float32, benchCols),
		bias:     make([]float32, benchCols),
		aFloat:   make([]float32, benchRows*benchK),
		wFloat:   make([]float32, benchCols*benchK),
		outFloat: make([]float32, benchRows*benchCols),
		outInt8:  make([]float32, benchRows*benchCols),
	}
	state := uint32(0x7f4a7c15)
	next := func() uint32 {
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5
		return state
	}
	for i := range d.aQ {
		q := uint8(next() >> 24)
		d.aQ[i] = q
		d.aFloat[i] = (float32(q) - float32(probeActivationZero)) * d.aScale
	}
	for n := 0; n < benchCols; n++ {
		// These are representative of the official checkpoint's per-output
		// scales; the checkpoint range for its first 384x1536 FC is
		// approximately 4.51e-4 to 3.06e-3.
		d.wScale[n] = 0.00045 + float32((n*7919)%26000)*0.0000001
		d.bias[n] = float32(int(n%17)-8) * 0.003
		for k := 0; k < benchK; k++ {
			// The official checkpoint uses symmetric QInt8 weights with
			// qmin=-127, qmax=127; paired Int16 accumulation depends on that.
			q := int8(int32(next()%255) - 127)
			index := n*benchK + k
			d.wQ[index] = q
			d.wFloat[index] = float32(q) * d.wScale[n]
		}
	}
	packWeightsKMajor(d.wQ, d.wPacked, d.wSum, benchCols, benchK)
	if err := d.checkFirstTile(); err != nil {
		panic(err)
	}
	return d
}

func (d *probeData) checkFirstTile() error {
	// Validate the signed-centering correction and all four SIMD lane groups
	// on one row before timing. Keep this scalar check small and outside the
	// benchmark interval.
	for n := 0; n < 32; n++ {
		var expected float32
		var integer int32
		for k := 0; k < benchK; k++ {
			expected += d.aFloat[k] * d.wFloat[n*benchK+k]
			centered := int32(int(d.aQ[k]) - int(probeActivationZero))
			integer += centered * int32(d.wQ[n*benchK+k])
		}
		expected += d.bias[n]
		got := float32(integer)*d.aScale*d.wScale[n] + d.bias[n]
		if delta := abs32(expected - got); delta > 0.0001 {
			return &mismatchError{column: n, expected: expected, got: got, delta: delta}
		}
	}
	// Check the vector kernel's 16-output lane arrangement with one small tile.
	const checkCols = 16
	checkW := make([]int8, checkCols*benchK)
	copy(checkW, d.wQ[:len(checkW)])
	checkPacked := make([]int8, len(checkW))
	checkSums := make([]int32, checkCols)
	checkScales := d.wScale[:checkCols]
	checkBias := d.bias[:checkCols]
	checkOutput := make([]float32, checkCols)
	packWeightsKMajor(checkW, checkPacked, checkSums, checkCols, benchK)
	quantizedMatMulKernel(
		d.aQ[:benchK], checkPacked, checkSums, d.aScale, probeActivationZero,
		checkScales, checkBias, checkOutput, 1, benchK, checkCols,
	)
	for n, got := range checkOutput {
		var expected float32
		for k := 0; k < benchK; k++ {
			expected += d.aFloat[k] * d.wFloat[n*benchK+k]
		}
		expected += d.bias[n]
		if delta := abs32(expected - got); delta > 0.0001 {
			return &mismatchError{column: n, expected: expected, got: got, delta: delta}
		}
	}
	return nil
}

type mismatchError struct {
	column        int
	expected, got float32
	delta         float32
}

func (e *mismatchError) Error() string {
	return "quantized scalar parity mismatch"
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func BenchmarkGEMMFP32_400x384x1536(b *testing.B) {
	d := makeProbeData()
	b.ReportAllocs()
	b.SetBytes(int64(benchRows*benchK*4 + benchCols*benchK*4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		float32MatMul(d.aFloat, d.wFloat, d.bias, d.outFloat)
	}
	probeResult = d.outFloat[len(d.outFloat)-1]
}

func BenchmarkGEMMINT8_400x384x1536(b *testing.B) {
	d := makeProbeData()
	b.ReportAllocs()
	b.SetBytes(int64(benchRows*benchK + benchCols*benchK + benchCols*4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quantizedMatMul(d.aQ, d.wPacked, d.wSum, d.aScale, probeActivationZero, d.wScale, d.bias, d.outInt8)
	}
	probeResult = d.outInt8[len(d.outInt8)-1]
}
