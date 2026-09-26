// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gpuportable

import (
	"errors"
	"fmt"
	"strings"
)

const linearLogicalTileSize = uint32(32)

type linearKernelKey struct {
	format           LinearFormat
	workgroupSize    uint32
	rowsPerWorkgroup uint32
	vectorized       bool
}

func defaultLinearGeometry(e *Engine, cols int) (uint32, uint32, error) {
	if e == nil || e.device == nil {
		return 0, 0, errors.New("gpuportable: nil engine")
	}
	limit := min(e.limits.MaxComputeWorkgroupSizeX, e.limits.MaxComputeInvocationsPerWorkgroup)
	// The matched Apple Q8B stream favors 128 over 256 work-items; cap down
	// only when an adapter reports a smaller supported workgroup.
	wg := uint32(128)
	for wg > limit {
		wg >>= 1
	}
	if wg < 32 {
		return 0, 0, fmt.Errorf("gpuportable: adapter workgroup limit %d is too small", limit)
	}
	rows := uint32(2)
	if cols%16 != 0 {
		rows = 1
	}
	return wg, rows, nil
}

func linearWGSL(format LinearFormat, workgroupSize, rowsPerWorkgroup uint32, vectorized bool) string {
	if !vectorized {
		return linearScalarBF16WGSL(workgroupSize)
	}
	var b strings.Builder
	if format == LinearBF16 {
		b.WriteString(`@group(0) @binding(0) var<storage, read> packedWeights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
fn weight4(a: u32, b: u32) -> vec4<f32> {
  let bits = vec4<u32>(a & 0xffffu, a >> 16u, b & 0xffffu, b >> 16u);
  return bitcast<vec4<f32>>(bits << vec4<u32>(16u));
}
`)
	} else if format == LinearQ8B {
		b.WriteString(`@group(0) @binding(0) var<storage, read> packedWeights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> packedScales: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
fn weight4(word: u32) -> vec4<f32> {
  let a = bitcast<i32>(word << 24u) >> 24u;
  let b = bitcast<i32>(word << 16u) >> 24u;
  let c = bitcast<i32>(word << 8u) >> 24u;
  let d = bitcast<i32>(word) >> 24u;
  return vec4<f32>(f32(a), f32(b), f32(c), f32(d));
}
fn scaleAt(i: u32) -> f32 {
  let pair = unpack2x16float(packedScales[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}
`)
	} else {
		return ""
	}

	tilesPerWorkgroup := workgroupSize / linearLogicalTileSize
	partialCount := workgroupSize * rowsPerWorkgroup
	fmt.Fprintf(&b, "var<workgroup> partial: array<f32, %d>;\n", partialCount)
	fmt.Fprintf(&b, `
@compute @workgroup_size(%d)
fn gemv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lane: vec3<u32>) {
  let rows = params[0];
  let cols = params[1];
  let rowBase = params[2];
  let tile = lane.x / %du;
  let tileLane = lane.x %% %du;
  let firstRow = wg.x * %du + tile * %du;
`, workgroupSize, linearLogicalTileSize, linearLogicalTileSize, tilesPerWorkgroup*rowsPerWorkgroup, rowsPerWorkgroup)
	for row := uint32(0); row < rowsPerWorkgroup; row++ {
		fmt.Fprintf(&b, "  var sum%d = 0.0;\n", row)
	}
	fmt.Fprintf(&b, `  for (var chunk = tileLane; chunk < cols / 16u; chunk += %du) {
    let activation0 = input[chunk * 4u];
    let activation1 = input[chunk * 4u + 1u];
    let activation2 = input[chunk * 4u + 2u];
    let activation3 = input[chunk * 4u + 3u];
`, linearLogicalTileSize)
	for row := uint32(0); row < rowsPerWorkgroup; row++ {
		fmt.Fprintf(&b, `    let row%d = firstRow + %du;
    if (row%d < rows) {
      let weightIndex%d = row%d * cols + chunk * 16u;
`, row, row, row, row, row)
		if format == LinearQ8B {
			fmt.Fprintf(&b, `      let packed%d = packedWeights[weightIndex%d >> 4u];
      let blocks = cols / 32u;
      let scale%d = scaleAt(row%d * blocks + (chunk >> 1u));
      sum%d += (dot(weight4(packed%d.x), activation0) + dot(weight4(packed%d.y), activation1) + dot(weight4(packed%d.z), activation2) + dot(weight4(packed%d.w), activation3)) * scale%d;
`, row, row, row, row, row, row, row, row, row, row)
		} else {
			fmt.Fprintf(&b, `      let packedA%d = packedWeights[weightIndex%d >> 3u];
      let packedB%d = packedWeights[(weightIndex%d >> 3u) + 1u];
      sum%d += dot(weight4(packedA%d.x, packedA%d.y), activation0) + dot(weight4(packedA%d.z, packedA%d.w), activation1) + dot(weight4(packedB%d.x, packedB%d.y), activation2) + dot(weight4(packedB%d.z, packedB%d.w), activation3);
`, row, row, row, row, row, row, row, row, row, row, row, row, row)
		}
		b.WriteString("    }\n")
	}
	b.WriteString("  }\n")
	for row := uint32(0); row < rowsPerWorkgroup; row++ {
		fmt.Fprintf(&b, "  partial[(tile * %du + %du) * %du + tileLane] = sum%d;\n", rowsPerWorkgroup, row, linearLogicalTileSize, row)
	}
	b.WriteString("  workgroupBarrier();\n")
	fmt.Fprintf(&b, "  var stride = %du;\n  loop {\n    if (stride == 0u) { break; }\n    if (tileLane < stride) {\n", linearLogicalTileSize/2)
	for row := uint32(0); row < rowsPerWorkgroup; row++ {
		fmt.Fprintf(&b, "      let slot%d = tile * %du + %du;\n      partial[slot%d * %du + tileLane] += partial[slot%d * %du + tileLane + stride];\n", row, rowsPerWorkgroup, row, row, linearLogicalTileSize, row, linearLogicalTileSize)
	}
	b.WriteString("    }\n    workgroupBarrier();\n    stride = stride >> 1u;\n  }\n  if (tileLane == 0u) {\n")
	for row := uint32(0); row < rowsPerWorkgroup; row++ {
		fmt.Fprintf(&b, "    let row%d = firstRow + %du;\n    if (row%d < rows) { output[rowBase + row%d] = partial[(tile * %du + %du) * %du]; }\n", row, row, row, row, rowsPerWorkgroup, row, linearLogicalTileSize)
	}
	b.WriteString("  }\n}\n")
	return b.String()
}

func linearScalarBF16WGSL(workgroupSize uint32) string {
	return fmt.Sprintf(`
@group(0) @binding(0) var<storage, read> packedWeights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
var<workgroup> partial: array<f32, %d>;
fn weightAt(i: u32) -> f32 {
  let word = packedWeights[i >> 1u];
  let bits = (word >> ((i & 1u) * 16u)) & 0xffffu;
  return bitcast<f32>(bits << 16u);
}
@compute @workgroup_size(%d)
fn gemv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lane: vec3<u32>) {
  let row = wg.x;
  let rows = params[0];
  let cols = params[1];
  let rowBase = params[2];
  if (row >= rows) { return; }
  var sum = 0.0;
  for (var col = lane.x; col < cols; col += %du) {
    sum += weightAt(row * cols + col) * input[col];
  }
  partial[lane.x] = sum;
  workgroupBarrier();
  var stride = %du;
  loop {
    if (stride == 0u) { break; }
    if (lane.x < stride) { partial[lane.x] += partial[lane.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lane.x == 0u) { output[rowBase + row] = partial[0]; }
}
`, workgroupSize, workgroupSize, workgroupSize, workgroupSize/2)
}
