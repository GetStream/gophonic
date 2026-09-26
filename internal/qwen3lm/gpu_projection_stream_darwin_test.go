// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/GetStream/gophonic/internal/metal"
	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/safetensors"
)

// BenchmarkGPUQ8BProjectionStream matches the portable stream's 28 distinct
// 2048x2048 Q8B matrices, nonzero inputs, one submit, and one final host read.
// Every byte is initialized: untouched zero buffers are not a bandwidth
// reference because page aliasing/compression may affect their behavior.
func BenchmarkGPUQ8BProjectionStream(b *testing.B) {
	dev, err := metal.Open()
	if err != nil {
		b.Skip(err)
	}
	defer dev.Close()
	lib, err := dev.Compile(gpuSource)
	if err != nil {
		b.Fatal(err)
	}
	kernel, err := dev.Pipeline(lib, "gemv_head") // Plain Q8B projection, no residual.
	if err != nil {
		b.Fatal(err)
	}
	const matrices, rows, cols = 28, 2048, 2048
	buffer := func(size int) *metal.Buffer {
		buf, err := dev.Buffer(size)
		if err != nil {
			b.Fatal(err)
		}
		return buf
	}
	x := buffer(4 * cols)
	defer x.Release()
	for i := range floats(x.Bytes()) {
		floats(x.Bytes())[i] = float32((i%31)-15) * 0.017
	}
	var weights, scales, output [matrices]*metal.Buffer
	for matrix := range matrices {
		weights[matrix] = buffer(rows * cols)
		defer weights[matrix].Release()
		scales[matrix] = buffer(rows * cols / 16)
		defer scales[matrix].Release()
		output[matrix] = buffer(rows * 4)
		defer output[matrix].Release()
		codes, scaleBytes := weights[matrix].Bytes(), scales[matrix].Bytes()
		// Same deterministic weight generator as the portable stream.
		for row := range rows {
			for block := range cols / 32 {
				var maxAbs float32
				for i := range 32 {
					v := float32((((row+matrix*13)*3+(block*32+i)*7)%101)-50) * 0.001
					maxAbs = max(maxAbs, abs32(v))
				}
				scale := safetensors.F16ToF32(q8gemm.F32ToF16(maxAbs / 127))
				binary.LittleEndian.PutUint16(scaleBytes[(row*(cols/32)+block)*2:], q8gemm.F32ToF16(scale))
				for i := range 32 {
					v := float32((((row+matrix*13)*3+(block*32+i)*7)%101)-50) * 0.001
					codes[row*cols+block*32+i] = byte(int8(math.RoundToEven(float64(v / scale))))
				}
			}
		}
	}
	var enc metal.Encoder
	args := gemvArgs{k: cols, n: rows}
	result := make([]float32, rows)
	run := func() {
		dev.Begin(&enc, false)
		enc.SetPipeline(kernel)
		enc.SetBuffer(x, 0, 2)
		enc.SetBuffer(output[0], 0, 4) // Unused by the plain projection.
		enc.SetBuffer(output[0], 0, 6)
		enc.SetBytes(unsafe.Pointer(&args), 16, 5)
		for matrix := range matrices {
			enc.SetBuffer(weights[matrix], 0, 0)
			enc.SetBuffer(scales[matrix], 0, 1)
			enc.SetBuffer(output[matrix], 0, 3)
			enc.Dispatch(metal.Size{X: rows / gpuHeadRows, Y: 1, Z: 1}, metal.Size{X: gpuThreads, Y: 1, Z: 1})
		}
		if err := enc.Wait(); err != nil {
			b.Fatal(err)
		}
		copy(result, floats(output[matrices-1].Bytes()))
	}
	for range 3 {
		run()
	}
	lastWeights, lastScales := weights[matrices-1].Bytes(), scales[matrices-1].Bytes()
	for row, got := range result {
		var want float64
		for col, value := range floats(x.Bytes()) {
			scaleBits := binary.LittleEndian.Uint16(lastScales[(row*(cols/32)+col/32)*2:])
			weight := float64(int8(lastWeights[row*cols+col])) * float64(safetensors.F16ToF32(scaleBits))
			want += weight * float64(value)
		}
		if !finite32(got) || math.Abs(float64(got)-want) > 1e-4*(1+math.Abs(want)) {
			b.Fatalf("projection row %d = %g, independent reference %g", row, got, want)
		}
	}
	b.SetBytes(matrices * rows * cols * 17 / 16)
	b.ReportAllocs()
	for b.Loop() {
		run()
	}
	b.ReportMetric(float64(matrices), "projections/op")
}
