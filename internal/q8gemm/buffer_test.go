// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package q8gemm

import (
	"math"
	"slices"
	"testing"
)

func TestWeightsF16BufferMatchesOwnedStorage(t *testing.T) {
	for _, shape := range [][2]int{{0, 0}, {0, 7}, {1, 1}, {3, 65}, {32, 129}} {
		k, n := shape[0], shape[1]
		size, err := WeightsF16Bytes(k, n)
		if err != nil {
			t.Fatal(err)
		}
		// Canaries on both sides detect writes beyond the borrowed region.
		data := make([]byte, size+8)
		for i := range data {
			data[i] = 0xa5
		}
		clear(data[4 : 4+size])
		got, err := NewWeightsF16Buffer(k, n, data[4:4+size:4+size])
		if err != nil {
			t.Fatal(err)
		}
		want, _ := NewWeightsF16(k, n)
		raw := make([]uint16, k*n)
		for i := range raw {
			raw[i] = uint16(math.Float32bits(float32(i%29-14)*0.125) >> 16)
		}
		if _, err := got.PackBF16(raw); err != nil {
			t.Fatal(err)
		}
		if _, err := want.PackBF16(raw); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got.h, want.h) || !slices.Equal(got.scales, want.scales) {
			t.Fatalf("shape %v differs", shape)
		}
		for _, b := range append(slices.Clone(data[:4]), data[4+size:]...) {
			if b != 0xa5 {
				t.Fatal("overwrote region guard")
			}
		}
	}
}

func TestWeightsF16BufferRejectsInvalidStorage(t *testing.T) {
	for _, shape := range [][2]int{{-1, 1}, {1, -1}, {math.MaxInt, 1}, {2, math.MaxInt}, {math.MaxInt / 2, 64}} {
		if _, err := WeightsF16Bytes(shape[0], shape[1]); err != ErrDimensions {
			t.Fatalf("shape %v: %v", shape, err)
		}
	}
	size, _ := WeightsF16Bytes(3, 65)
	data := make([]byte, size+4)
	if _, err := NewWeightsF16Buffer(3, 65, data[:size-1]); err != ErrDimensions {
		t.Fatal("accepted short storage")
	}
	if _, err := NewWeightsF16Buffer(3, 65, data[1:]); err != ErrDimensions {
		t.Fatal("accepted misaligned storage")
	}
}
