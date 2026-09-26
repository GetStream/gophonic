// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gpuportable

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/gogpu/wgpu"
)

// BenchmarkLinearProjectionStream compares portable WGSL projection geometry
// over the same 28 distinct 2048x2048 matrices used by the native ASR O-proj
// stream. Each iteration uploads no activations, submits all 28 dispatches
// once, and maps one result buffer once.
func BenchmarkLinearProjectionStream(b *testing.B) {
	if os.Getenv("GOPHONIC_GPU_BENCH") != "1" {
		b.Skip("set GOPHONIC_GPU_BENCH=1 to run GPU throughput qualification")
	}
	e, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	b.Logf("adapter=%q backend=%s device=%s workgroupX=%d invocations=%d", e.AdapterInfo().Name, e.AdapterInfo().Backend, e.AdapterInfo().DeviceType, e.Limits().MaxComputeWorkgroupSizeX, e.Limits().MaxComputeInvocationsPerWorkgroup)
	const matrices, rows, cols = 28, 2048, 2048
	input := make([]float32, cols)
	for i := range input {
		input[i] = float32((i%31)-15) * 0.017
	}
	geometries := []struct {
		wg, rps uint32
	}{
		{128, 1}, // packed one-row baseline
		{32, 2},
		{32, 4},
		{64, 2},
		{128, 2},
		{64, 4},
		{128, 4},
		{256, 2},
		{256, 4},
	}
	formats := []struct {
		name   string
		format LinearFormat
		bytes  int64
	}{
		{"BF16", LinearBF16, int64(matrices * rows * cols * 2)},
		{"Q8B", LinearQ8B, int64(matrices * rows * cols * 17 / 16)},
	}
	for _, f := range formats {
		for _, g := range geometries {
			b.Run(fmt.Sprintf("%s/wg%d-r%d", f.name, g.wg, g.rps), func(b *testing.B) {
				benchmarkLinearStream(b, e, f.format, matrices, rows, cols, input, g.wg, g.rps, f.bytes, true, true)
			})
		}
	}
}

// BenchmarkLinearProjectionBankStream packs same-input projections into one
// row-major matrix and dispatches them together. It models portable QKV or
// Gate+Up fusion and the 28-matrix stream while measuring launch and binding
// overhead separately from the weight bytes read by the GPU.
func BenchmarkLinearProjectionBankStream(b *testing.B) {
	if os.Getenv("GOPHONIC_GPU_BENCH") != "1" {
		b.Skip("set GOPHONIC_GPU_BENCH=1 to run GPU throughput qualification")
	}
	e, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	const matrices, rows, cols = 28, 2048, 2048
	input := make([]float32, cols)
	for i := range input {
		input[i] = float32((i%31)-15) * 0.017
	}
	weights := make([]byte, 0, matrices*rows*cols)
	scales := make([]byte, 0, matrices*rows*(cols/32)*2)
	for matrix := range matrices {
		w, s := benchmarkPackedWeights(LinearQ8B, rows, cols, matrix)
		weights = append(weights, w...)
		scales = append(scales, s...)
	}
	linear, err := newLinearGeometry(e, LinearQ8B, matrices*rows, cols, weights, scales, 16<<20, 128, 2)
	if err != nil {
		b.Fatal(err)
	}
	defer linear.Close()
	lane, err := linear.NewWorkspace()
	if err != nil {
		b.Fatal(err)
	}
	defer lane.Close()
	inputBytes := make([]byte, cols*4)
	for i, value := range input {
		binary.LittleEndian.PutUint32(inputBytes[i*4:], math.Float32bits(value))
	}
	if err := e.Upload(lane.input, 0, inputBytes); err != nil {
		b.Fatal(err)
	}
	if err := e.Submit(); err != nil {
		b.Fatal(err)
	}
	if err := e.device.WaitIdle(); err != nil {
		b.Fatal(err)
	}
	if len(lane.dispatches) != 1 {
		b.Logf("matrix bank uses %d dispatches due adapter binding limits", len(lane.dispatches))
	}
	result := make([]byte, rows*4)
	lastOffset := uint64((matrices - 1) * rows * 4)
	b.SetBytes(int64(matrices * rows * cols * 17 / 16))
	b.ReportMetric(float64(matrices), "projections/op")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err = e.SubmitReadback(lane.dispatches, lane.output, lastOffset, result, &lane.staging)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLinearProjectionStreamNoReadback(b *testing.B) {
	if os.Getenv("GOPHONIC_GPU_BENCH") != "1" {
		b.Skip("set GOPHONIC_GPU_BENCH=1 to run GPU throughput qualification")
	}
	e, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	const matrices, rows, cols = 28, 2048, 2048
	input := make([]float32, cols)
	for i := range input {
		input[i] = float32((i%31)-15) * 0.017
	}
	bytes := int64(matrices * rows * cols * 17 / 16)
	benchmarkLinearStream(b, e, LinearQ8B, matrices, rows, cols, input, 128, 2, bytes, false, true)
}

func BenchmarkLinearProjectionSubmitCPU(b *testing.B) {
	if os.Getenv("GOPHONIC_GPU_BENCH") != "1" {
		b.Skip("set GOPHONIC_GPU_BENCH=1 to run GPU throughput qualification")
	}
	if b.N > 1024 {
		b.Skip("run with -benchtime=1024x or less to bound outstanding GPU submissions")
	}
	e, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	const matrices, rows, cols = 28, 2048, 2048
	input := make([]float32, cols)
	for i := range input {
		input[i] = float32((i%31)-15) * 0.017
	}
	bytes := int64(matrices * rows * cols * 17 / 16)
	benchmarkLinearStream(b, e, LinearQ8B, matrices, rows, cols, input, 64, 2, bytes, false, false)
}

func benchmarkLinearStream(b *testing.B, e *Engine, format LinearFormat, matrices, rows, cols int, input []float32, wg, rps uint32, bytes int64, readback, waitEach bool) {
	b.Helper()
	sharedInput, err := e.NewBuffer("gophonic-benchmark-input", uint64(cols*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		b.Fatal(err)
	}
	defer sharedInput.Release()
	linears := make([]*Linear, 0, matrices)
	lanes := make([]*LinearWorkspace, 0, matrices)
	defer func() {
		for _, lane := range lanes {
			lane.Close()
		}
		for _, linear := range linears {
			linear.Close()
		}
	}()
	for matrix := range matrices {
		weights, scales := benchmarkPackedWeights(format, rows, cols, matrix)
		linear, err := newLinearGeometry(e, format, rows, cols, weights, scales, 16<<20, wg, rps)
		if err != nil {
			if err.Error() == fmt.Sprintf("gpuportable: unsupported linear geometry workgroup=%d rows=%d", wg, rps) {
				b.Skipf("adapter does not support WGSL geometry %d/%d: %v", wg, rps, err)
			}
			b.Fatal(err)
		}
		linears = append(linears, linear)
		lane, err := linear.newWorkspace(sharedInput, nil)
		if err != nil {
			b.Fatal(err)
		}
		lanes = append(lanes, lane)
		weights, scales = nil, nil
	}
	inputBytes := make([]byte, cols*4)
	for i, value := range input {
		binary.LittleEndian.PutUint32(inputBytes[i*4:], math.Float32bits(value))
	}
	if err := e.Upload(sharedInput, 0, inputBytes); err != nil {
		b.Fatal(err)
	}
	if err := e.Submit(); err != nil {
		b.Fatal(err)
	}
	if err := e.device.WaitIdle(); err != nil {
		b.Fatal(err)
	}
	dispatches := make([]Dispatch, 0, matrices)
	for _, lane := range lanes {
		dispatches = append(dispatches, lane.dispatches...)
	}
	result := make([]byte, rows*4)
	last := lanes[len(lanes)-1]
	b.SetBytes(bytes)
	b.ReportMetric(float64(matrices), "projections/op")
	b.ReportAllocs()
	b.ResetTimer()
	for i := range b.N {
		if readback {
			err = e.SubmitIndependentReadback(dispatches, last.output, 0, result, &last.staging)
		} else {
			err = e.SubmitIndependent(dispatches...)
			if err == nil && waitEach {
				err = e.device.WaitIdle()
			}
		}
		if err != nil {
			b.Fatal(err)
		}
		if !readback && !waitEach && (i+1)%32 == 0 && i+1 < b.N {
			b.StopTimer()
			if err := e.device.WaitIdle(); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
		}
	}
	if !readback && !waitEach {
		b.StopTimer()
		if err := e.device.WaitIdle(); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkPackedWeights(format LinearFormat, rows, cols, seed int) (weights, scales []byte) {
	if format == LinearBF16 {
		weights = make([]byte, rows*cols*2)
		for row := range rows {
			for col := 0; col < cols; col++ {
				v := float32((((row+seed*13)*3+col*7)%101)-50) * 0.001
				binary.LittleEndian.PutUint16(weights[(row*cols+col)*2:], uint16(math.Float32bits(v)>>16))
			}
		}
		return weights, nil
	}
	weights = make([]byte, rows*cols)
	blocks := cols / 32
	scales = make([]byte, rows*blocks*2)
	for row := range rows {
		for block := range blocks {
			var maxAbs float32
			for i := range 32 {
				v := float32((((row+seed*13)*3+(block*32+i)*7)%101)-50) * 0.001
				maxAbs = max(maxAbs, float32(math.Abs(float64(v))))
			}
			scale := safetensors.F16ToF32(q8gemm.F32ToF16(maxAbs / 127))
			binary.LittleEndian.PutUint16(scales[(row*blocks+block)*2:], q8gemm.F32ToF16(scale))
			for i := range 32 {
				v := float32((((row+seed*13)*3+(block*32+i)*7)%101)-50) * 0.001
				q := int8(math.RoundToEven(float64(v / scale)))
				weights[row*cols+block*32+i] = byte(q)
			}
		}
	}
	return weights, scales
}
