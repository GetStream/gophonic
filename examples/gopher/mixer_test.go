// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import "testing"

// A late packet is a gap, heard in order when it arrives; a microphone
// that stops sending is silence after maxGap frames.
func TestMixerGaps(t *testing.T) {
	m := newMixer()
	frame := func(v float32) []float32 {
		f := make([]float32, 320)
		for i := range f {
			f[i] = v
		}
		return f
	}
	dst := make([]float32, 320)
	m.write("a", frame(1))
	if !m.read(dst) || dst[0] != 1 {
		t.Fatal("the first frame")
	}
	if m.read(dst) {
		t.Fatal("a late frame read as silence")
	}
	m.write("a", frame(2))
	m.write("a", frame(3))
	for _, want := range []float32{2, 3} {
		if !m.read(dst) || dst[0] != want {
			t.Fatalf("late audio out of order: %v, want %v", dst[0], want)
		}
	}
	for i := range maxGap {
		if m.read(dst) {
			t.Fatalf("silence after %d frames", i)
		}
	}
	if !m.read(dst) || dst[0] != 0 {
		t.Fatal("a stopped microphone is silence")
	}
}
