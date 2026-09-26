// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gpuportable

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/q8gemm"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

const bf16GemvWGSL = `
@group(0) @binding(0) var<storage, read> packedWeights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

fn weightAt(i: u32) -> f32 {
  let packed = packedWeights[i >> 1u];
  let shift = (i & 1u) * 16u;
  return bitcast<f32>(((packed >> shift) & 0xffffu) << 16u);
}

@compute @workgroup_size(64)
fn gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let rows = params[0];
  let cols = params[1];
  if (row >= rows) { return; }
  var sum = 0.0;
  for (var col = 0u; col < cols; col += 1u) {
    sum += weightAt(row * cols + col) * input[col];
  }
  output[row] = sum;
}
`

// Set GOPHONIC_GPU_TESTS=1 to make absence of an accelerated adapter fail,
// which turns this test into a hardware qualification gate in controlled CI.
func TestPackedBF16WGSLGemvParity(t *testing.T) {
	e := gpuTestEngine(t)
	defer e.Close()
	t.Logf("adapter=%q backend=%s device=%s limits(storage=%d,buffer=%d)", e.AdapterInfo().Name, e.AdapterInfo().Backend, e.AdapterInfo().DeviceType, e.Limits().MaxStorageBufferBindingSize, e.Limits().MaxBufferSize)

	const rows, cols = 7, 19
	bfloat := make([]uint16, rows*cols)
	input := make([]float32, cols)
	for i := range bfloat {
		v := float32((i%17)-8) * 0.03125
		bfloat[i] = uint16(math.Float32bits(v) >> 16)
	}
	for i := range input {
		input[i] = float32((i%9)-4) * 0.125
	}
	packed := make([]byte, ((len(bfloat)+1)/2)*4)
	for i, v := range bfloat {
		off := (i / 2) * 4
		shift := uint((i % 2) * 16)
		word := binary.LittleEndian.Uint32(packed[off : off+4])
		word |= uint32(v) << shift
		binary.LittleEndian.PutUint32(packed[off:off+4], word)
	}
	inputBytes := make([]byte, len(input)*4)
	for i, v := range input {
		binary.LittleEndian.PutUint32(inputBytes[i*4:], math.Float32bits(v))
	}
	paramBytes := make([]byte, 16)
	binary.LittleEndian.PutUint32(paramBytes[0:4], rows)
	binary.LittleEndian.PutUint32(paramBytes[4:8], cols)

	weights, err := e.NewBuffer("test-bf16-weights", uint64(len(packed)), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Release()
	inputBuffer, err := e.NewBuffer("test-input", uint64(len(inputBytes)), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		t.Fatal(err)
	}
	defer inputBuffer.Release()
	outputBuffer, err := e.NewBuffer("test-output", rows*4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	if err != nil {
		t.Fatal(err)
	}
	defer outputBuffer.Release()
	params, err := e.NewBuffer("test-params", uint64(len(paramBytes)), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		t.Fatal(err)
	}
	defer params.Release()
	for _, upload := range []struct {
		buffer *wgpu.Buffer
		data   []byte
	}{{weights, packed}, {inputBuffer, inputBytes}, {params, paramBytes}} {
		if err := e.Upload(upload.buffer, 0, upload.data); err != nil {
			t.Fatal(err)
		}
	}

	kernel, err := e.NewKernel("test-bf16-gemv", bf16GemvWGSL, "gemv", []Binding{
		BindingReadOnlyStorage, BindingReadOnlyStorage, BindingStorage, BindingReadOnlyStorage,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer kernel.Close()
	bind, err := kernel.Bind("test-bf16-gemv-bindings",
		BufferRange{Buffer: weights, Size: uint64(len(packed))},
		BufferRange{Buffer: inputBuffer, Size: uint64(len(inputBytes))},
		BufferRange{Buffer: outputBuffer, Size: rows * 4},
		BufferRange{Buffer: params, Size: uint64(len(paramBytes))},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer bind.Release()
	var staging *wgpu.Buffer
	defer func() {
		if staging != nil {
			staging.Release()
		}
	}()
	gotBytes := make([]byte, rows*4)
	if err := e.SubmitReadback([]Dispatch{{Kernel: kernel, BindGroup: bind, Workgroups: [3]uint32{1, 1, 1}}}, outputBuffer, 0, gotBytes, &staging); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < rows; row++ {
		var want float32
		for col := 0; col < cols; col++ {
			weight := math.Float32frombits(uint32(bfloat[row*cols+col]) << 16)
			want += weight * input[col]
		}
		got := math.Float32frombits(binary.LittleEndian.Uint32(gotBytes[row*4:]))
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Abs(float64(got-want)) > 1e-6*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("row %d: got %.9g, want %.9g", row, got, want)
		}
	}
}

func gpuTestEngine(t *testing.T) *Engine {
	t.Helper()
	e, err := New()
	if err != nil {
		if os.Getenv("GOPHONIC_GPU_TESTS") == "1" {
			t.Fatal(err)
		}
		t.Skipf("no accelerated WebGPU adapter: %v", err)
	}
	if os.Getenv("GOPHONIC_GPU_REQUIRE_VULKAN") == "1" && e.AdapterInfo().Backend != gputypes.BackendVulkan {
		info := e.AdapterInfo()
		e.Close()
		t.Fatalf("GPU qualification requires Vulkan, got backend=%s adapter=%q device=%s", info.Backend, info.Name, info.DeviceType)
	}
	return e
}

func TestRejectsSoftwareAdapterMetadata(t *testing.T) {
	for _, info := range []gputypes.AdapterInfo{
		{Backend: gputypes.BackendVulkan, DeviceType: gputypes.DeviceTypeCPU},
		{Backend: gputypes.BackendGL, DeviceType: gputypes.DeviceTypeIntegratedGPU},
		{Backend: gputypes.BackendEmpty, DeviceType: gputypes.DeviceTypeOther},
	} {
		if !acceleratedAdapter(info, false) {
			t.Logf("rejected software/unsupported adapter metadata: %s/%s", info.Backend, info.DeviceType)
		} else {
			t.Errorf("accepted software/unsupported adapter metadata: %s/%s", info.Backend, info.DeviceType)
		}
	}
	if !acceleratedAdapter(gputypes.AdapterInfo{Backend: gputypes.BackendVulkan, DeviceType: gputypes.DeviceTypeCPU}, true) {
		t.Error("explicit Vulkan CPU test adapter should be allowed")
	}
}

func TestLinearBF16AndLaneIsolation(t *testing.T) {
	e := gpuTestEngine(t)
	defer e.Close()
	const rows, cols = 17, 256
	bf16 := make([]uint16, rows*cols)
	weights := make([]byte, rows*cols*2)
	for i := range bf16 {
		v := float32((i%47)-23) * 0.002
		bf16[i] = uint16(math.Float32bits(v) >> 16)
		binary.LittleEndian.PutUint16(weights[i*2:], bf16[i])
	}
	linear, err := NewLinear(e, LinearBF16, rows, cols, weights, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer linear.Close()
	laneA, err := linear.NewWorkspace()
	if err != nil {
		t.Fatal(err)
	}
	defer laneA.Close()
	laneB, err := linear.NewWorkspace()
	if err != nil {
		t.Fatal(err)
	}
	defer laneB.Close()
	if laneA.input == laneB.input || laneA.output == laneB.output || laneA.params == laneB.params {
		t.Fatal("linear lanes share mutable device buffers")
	}
	inputA, inputB := make([]float32, cols), make([]float32, cols)
	for i := range cols {
		inputA[i] = float32((i%13)-6) * 0.03125
		inputB[i] = float32((i%7)-3) * 0.0625
	}
	for _, pair := range []struct {
		lane *LinearWorkspace
		in   []float32
	}{{laneA, inputA}, {laneB, inputB}} {
		got := make([]float32, rows)
		if err := pair.lane.Run(pair.in, got); err != nil {
			t.Fatal(err)
		}
		want := make([]float32, rows)
		for row := range rows {
			for col := range cols {
				weight := math.Float32frombits(uint32(bf16[row*cols+col]) << 16)
				want[row] += weight * pair.in[col]
			}
		}
		assertLinearClose(t, got, want, 2e-5)
	}
}

func TestLinearQ8BParity(t *testing.T) {
	e := gpuTestEngine(t)
	defer e.Close()
	const rows, cols = 11, 256
	const blocksPerRow = cols / 32
	source := make([]float32, rows*cols)
	weights := make([]byte, rows*cols)
	scales := make([]byte, rows*blocksPerRow*2)
	for i := range source {
		source[i] = float32((i%61)-30) * 0.003
	}
	for row := range rows {
		for block := range blocksPerRow {
			var maxAbs float32
			for i := range 32 {
				v := source[row*cols+block*32+i]
				maxAbs = max(maxAbs, float32(math.Abs(float64(v))))
			}
			scale := safetensors.F16ToF32(q8gemm.F32ToF16(maxAbs / 127))
			binary.LittleEndian.PutUint16(scales[(row*blocksPerRow+block)*2:], q8gemm.F32ToF16(scale))
			for i := range 32 {
				q := int8(math.RoundToEven(float64(source[row*cols+block*32+i] / scale)))
				weights[row*cols+block*32+i] = byte(q)
			}
		}
	}
	linear, err := NewLinear(e, LinearQ8B, rows, cols, weights, scales, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer linear.Close()
	lane, err := linear.NewWorkspace()
	if err != nil {
		t.Fatal(err)
	}
	defer lane.Close()
	input, got, want := make([]float32, cols), make([]float32, rows), make([]float32, rows)
	for i := range input {
		input[i] = float32((i%23)-11) * 0.027
	}
	for row := range rows {
		for col := range cols {
			q := int8(weights[row*cols+col])
			scaleIndex := (row*blocksPerRow + col/32) * 2
			scale := safetensors.F16ToF32(binary.LittleEndian.Uint16(scales[scaleIndex:]))
			want[row] += float32(q) * scale * input[col]
		}
	}
	if err := lane.Run(input, got); err != nil {
		t.Fatal(err)
	}
	assertLinearClose(t, got, want, 5e-5)
}

func TestDependentDispatchPassOrdering(t *testing.T) {
	e := gpuTestEngine(t)
	defer e.Close()
	const firstShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < 4u) { output[gid.x] = input[gid.x] * 2.0; }
}`
	const secondShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < 4u) { output[gid.x] = input[gid.x] + 3.0; }
}`
	bindings := []Binding{BindingReadOnlyStorage, BindingStorage}
	firstKernel, err := e.NewKernel("ordering-double", firstShader, "main", bindings)
	if err != nil {
		t.Fatal(err)
	}
	defer firstKernel.Close()
	secondKernel, err := e.NewKernel("ordering-add", secondShader, "main", bindings)
	if err != nil {
		t.Fatal(err)
	}
	defer secondKernel.Close()
	input, err := e.NewBuffer("ordering-input", 16, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Release()
	middle, err := e.NewBuffer("ordering-middle", 16, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		t.Fatal(err)
	}
	defer middle.Release()
	output, err := e.NewBuffer("ordering-output", 16, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Release()
	inputValues := []float32{1, 2, 3, 4}
	inputBytes := make([]byte, 16)
	for i, v := range inputValues {
		binary.LittleEndian.PutUint32(inputBytes[i*4:], math.Float32bits(v))
	}
	if err := e.Upload(input, 0, inputBytes); err != nil {
		t.Fatal(err)
	}
	first, err := firstKernel.Bind("ordering-first", BufferRange{Buffer: input, Size: 16}, BufferRange{Buffer: middle, Size: 16})
	if err != nil {
		t.Fatal(err)
	}
	defer first.Release()
	second, err := secondKernel.Bind("ordering-second", BufferRange{Buffer: middle, Size: 16}, BufferRange{Buffer: output, Size: 16})
	if err != nil {
		t.Fatal(err)
	}
	defer second.Release()
	var staging *wgpu.Buffer
	defer func() {
		if staging != nil {
			staging.Release()
		}
	}()
	gotBytes := make([]byte, 16)
	dispatches := []Dispatch{
		{Kernel: firstKernel, BindGroup: first, Workgroups: [3]uint32{1, 1, 1}},
		{Kernel: secondKernel, BindGroup: second, Workgroups: [3]uint32{1, 1, 1}},
	}
	if err := e.SubmitReadback(dispatches, output, 0, gotBytes, &staging); err != nil {
		t.Fatal(err)
	}
	for i, value := range inputValues {
		got := math.Float32frombits(binary.LittleEndian.Uint32(gotBytes[i*4:]))
		if got != 2*value+3 {
			t.Fatalf("element %d: got %g, want %g", i, got, 2*value+3)
		}
	}
}

func assertLinearClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-want[i])) > tol*math.Max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("value %d: got %.9g, want %.9g", i, value, want[i])
		}
	}
}
