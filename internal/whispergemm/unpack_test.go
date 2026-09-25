// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import (
	"math"
	"testing"
)

func TestUnpackRangesPreserveBitsAndPadding(t *testing.T) {
	for _, width := range []int{0, 1, 15, 16, 17, 24, 33} {
		const rows = 273
		stride := width + 3
		raw := make([]float32, rows*stride)
		for r := range rows {
			for c := range width {
				bits := uint32((r*37+c*53)%10000) | 0x3f000000
				if (r+c)%29 == 0 {
					bits = 0x80000000
				}
				if (r+c)%31 == 0 {
					bits = 0x7fc00001
				}
				raw[r*stride+c] = math.Float32frombits(bits)
			}
		}
		keys, _ := NewPackedB(width, rows)
		if err := keys.Pack(raw, stride, true); err != nil {
			t.Fatal(err)
		}
		size, _ := PackedLen(rows+19, width)
		values, _ := NewPackedBRows(rows+19, width, make([]float32, size))
		if err := values.PackRows(raw, stride, 0, rows); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, rows*stride)
		for _, span := range [][2]int{{0, rows}, {15, 17}, {16, 0}, {17, 128}, {127, 129}, {255, 18}, {rows, 0}} {
			for _, columns := range []bool{false, true} {
				for i := range out {
					out[i] = -777
				}
				start, n := span[0], span[1]
				var err error
				if columns {
					err = keys.UnpackColumns(out, stride, start, n)
				} else {
					err = values.UnpackRows(out, stride, start, n)
				}
				if err != nil {
					t.Fatal(err)
				}
				for r := range rows {
					for c := range stride {
						want := float32(-777)
						if r < n && c < width {
							want = raw[(start+r)*stride+c]
						}
						if math.Float32bits(out[r*stride+c]) != math.Float32bits(want) {
							t.Fatalf("width=%d span=%v columns=%v at (%d,%d)", width, span, columns, r, c)
						}
					}
				}
			}
		}
		for _, span := range [][2]int{{-1, 1}, {0, -1}, {rows, 1}, {1, math.MaxInt}} {
			if keys.UnpackColumns(out, stride, span[0], span[1]) != ErrShape || values.UnpackRows(out, stride, span[0], span[1]) != ErrShape {
				t.Fatalf("accepted invalid range %v", span)
			}
		}
		if n := testing.AllocsPerRun(1, func() {
			if err := keys.UnpackColumns(out, stride, 15, 129); err != nil {
				panic(err)
			}
			if err := values.UnpackRows(out, stride, 15, 129); err != nil {
				panic(err)
			}
		}); n != 0 {
			t.Fatalf("unpack allocations=%g", n)
		}
	}
	var nilMatrix *PackedB
	if nilMatrix.UnpackColumns(nil, 0, 0, 0) != ErrNilMatrix || nilMatrix.UnpackRows(nil, 0, 0, 0) != ErrNilMatrix {
		t.Fatal("nil matrix accepted")
	}
}
