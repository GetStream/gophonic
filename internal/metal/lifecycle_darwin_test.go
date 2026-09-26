// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"testing"
)

// The MTL library can be released after pipeline compilation. Inference
// retains only the immutable pipelines and explicitly releases them on close.
func TestCompiledResourceRelease(t *testing.T) {
	dev, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	lib, err := dev.Compile(`#include <metal_stdlib>
using namespace metal;
kernel void lifecycle(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = i * 3 + 7;
}`)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Release()
	pipeline, err := dev.Pipeline(lib, "lifecycle")
	if err != nil {
		t.Fatal(err)
	}
	defer pipeline.Release()
	lib.Release()
	lib.Release()
	if lib.lib != 0 {
		t.Fatal("released library retained its native handle")
	}
	out, err := dev.Buffer(4 * 32)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Release()
	var encoder Encoder
	dev.Begin(&encoder, false)
	encoder.SetPipeline(pipeline)
	encoder.SetBuffer(out, 0, 0)
	encoder.Dispatch(Size{X: 1, Y: 1, Z: 1}, Size{X: 32, Y: 1, Z: 1})
	if err := encoder.Wait(); err != nil {
		t.Fatal(err)
	}
	for i := range 32 {
		if got := binary.LittleEndian.Uint32(out.Bytes()[4*i:]); got != uint32(i*3+7) {
			t.Fatalf("output %d = %d", i, got)
		}
	}
	pipeline.Release()
	pipeline.Release()
	if pipeline.p != 0 {
		t.Fatal("released pipeline retained its native handle")
	}
	(*Library)(nil).Release()
	(*Pipeline)(nil).Release()
}
