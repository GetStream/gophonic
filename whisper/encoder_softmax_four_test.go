// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"fmt"
	"math"
	"testing"
)

func TestAudioSoftmaxGroupsPreserveRows(t *testing.T) {
	for _, rows := range []int{0, 1, 3, 4, 5, 8, 32} {
		for _, columns := range []int{1, 2, 3, 4, 5, 7, 16, 17, 65, 1500} {
			for offset := range 4 {
				t.Run(fmt.Sprintf("rows%d/columns%d/offset%d", rows, columns, offset), func(t *testing.T) {
					storage := make([]float32, offset+rows*columns+1)
					for i := range storage {
						storage[i] = -123456
					}
					values := storage[offset : offset+rows*columns]
					for row := range rows {
						for column := range columns {
							value := float32(30*math.Sin(float64(column)*0.13+float64(row)) - float64(row)*17)
							if row%4 == 1 {
								value = -float32(column % 111)
								if column%23 == 0 {
									value = float32(math.Inf(-1))
								}
							}
							if row%4 == 2 {
								value = float32(math.Copysign(0, float64(column%2)-0.5))
							}
							values[row*columns+column] = value
						}
					}
					want := append([]float32(nil), values...)
					for row := range rows {
						softmaxRow(want[row*columns : (row+1)*columns])
					}
					softmaxRows(values, rows, columns)
					for i, value := range values {
						if math.IsNaN(float64(want[i])) && math.IsNaN(float64(value)) {
							continue
						}
						if math.Float32bits(value) != math.Float32bits(want[i]) {
							t.Fatalf("output[%d]=%.9g (%08x), single-row=%.9g (%08x)", i, value, math.Float32bits(value), want[i], math.Float32bits(want[i]))
						}
					}
					for i, value := range storage {
						if (i < offset || i >= offset+len(values)) && value != -123456 {
							t.Fatalf("overwrote score padding at %d", i)
						}
					}
				})
			}
		}
	}
}

func TestAudioSoftmaxGroupsSpecialValues(t *testing.T) {
	for _, special := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		for _, columns := range []int{4, 5, 17} {
			for row := range 4 {
				values := make([]float32, 4*columns)
				for i := range values {
					values[i] = float32(i%11) - 5
				}
				values[(row+1)*columns-1] = special
				want := append([]float32(nil), values...)
				for r := range 4 {
					softmaxRow(want[r*columns : (r+1)*columns])
				}
				softmaxRows(values, 4, columns)
				for i, value := range values {
					if math.IsNaN(float64(want[i])) && math.IsNaN(float64(value)) {
						continue
					}
					if math.Float32bits(value) != math.Float32bits(want[i]) {
						t.Fatalf("special=%g columns=%d row=%d output[%d]=%.9g, want %.9g", special, columns, row, i, value, want[i])
					}
				}
			}
		}
	}
}

func TestAudioSoftmaxGroupsNoAlloc(t *testing.T) {
	const rows, columns = 32, 1500
	src, dst := make([]float32, rows*columns), make([]float32, rows*columns)
	for i := range src {
		src[i] = float32(math.Sin(float64(i)*0.07) * 8)
	}
	if allocations := testing.AllocsPerRun(5, func() {
		copy(dst, src)
		softmaxRows(dst, rows, columns)
	}); allocations != 0 {
		t.Fatalf("warm grouped softmax allocated %g objects", allocations)
	}
}
