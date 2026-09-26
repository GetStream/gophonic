// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"testing"
	"unsafe"
)

// Check actual device results after every transition that invalidates cached
// state: inline bytes, offset, buffer, pipeline, and a new command encoder.
func TestEncoderBindingTransitions(t *testing.T) {
	dev, err := Open()
	if err != nil {
		t.Skip(err)
	}
	t.Cleanup(dev.Close)
	lib, err := dev.Compile(`#include <metal_stdlib>
using namespace metal;
kernel void copy_value(device uint *x [[buffer(0)]], device uint *out [[buffer(1)]], constant uint &at [[buffer(2)]]) { out[at] = x[0]; }
kernel void plus_one(device uint *x [[buffer(0)]], device uint *out [[buffer(1)]], constant uint &at [[buffer(2)]]) { out[at] = x[0] + 1; }
`)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Release()
	p, err := dev.Pipeline(lib, "copy_value")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Release()
	plus, err := dev.Pipeline(lib, "plus_one")
	if err != nil {
		t.Fatal(err)
	}
	defer plus.Release()
	buffer := func(n int) *Buffer {
		t.Helper()
		b, err := dev.Buffer(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Release)
		return b
	}
	x, other, out := buffer(8), buffer(4), buffer(28)
	binary.LittleEndian.PutUint32(x.Bytes(), 11)
	binary.LittleEndian.PutUint32(x.Bytes()[4:], 22)
	binary.LittleEndian.PutUint32(other.Bytes(), 44)
	for _, concurrent := range []bool{false, true} {
		var e Encoder
		var at uint32
		emit := func() {
			e.SetBytes(unsafe.Pointer(&at), 4, 2)
			e.Dispatch(Size{X: 1, Y: 1, Z: 1}, Size{X: 1, Y: 1, Z: 1})
			at++
		}
		dev.Begin(&e, concurrent)
		e.SetPipeline(p)
		e.SetPipeline(p)
		e.SetBuffer(x, 0, 0)
		e.SetBuffer(x, 0, 0)
		e.SetBuffer(out, 0, 1)
		emit()
		inline := uint32(33)
		e.SetBytes(unsafe.Pointer(&inline), 4, 0)
		emit()
		e.SetBuffer(x, 0, 0)
		emit()
		e.SetBuffer(x, 4, 0)
		emit()
		e.SetPipeline(plus)
		emit()
		if err := e.Wait(); err != nil {
			t.Fatal(err)
		}
		dev.Begin(&e, concurrent)
		e.SetPipeline(plus)
		e.SetBuffer(x, 4, 0)
		e.SetBuffer(out, 0, 1)
		emit()
		e.SetBuffer(other, 0, 0)
		emit()
		if err := e.Wait(); err != nil {
			t.Fatal(err)
		}
		for i, want := range []uint32{11, 33, 11, 22, 23, 23, 45} {
			if got := binary.LittleEndian.Uint32(out.Bytes()[4*i:]); got != want {
				t.Fatalf("concurrent=%v output %d: got %d, want %d", concurrent, i, got, want)
			}
		}
	}
}
