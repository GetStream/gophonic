// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whisper

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestAudioSoftmaxGroupsPinnedNEONOracle(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "softmax_exp_neon.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 64*32 {
		t.Fatalf("NEON fixture size=%d, want %d", len(data), 64*32)
	}
	// Each row ends in four zeros, making its maximum zero unless it contains
	// NaN. The stored C exponential results therefore determine every output
	// directly, including normal values, subnormals, infinities and NaNs.
	const rows, columns = 4, 68
	values, want := make([]float32, rows*columns), make([]float32, rows*columns)
	for record := range 64 {
		for row := range rows {
			values[row*columns+record] = math.Float32frombits(binary.LittleEndian.Uint32(data[record*32+row*4:]))
			want[row*columns+record] = math.Float32frombits(binary.LittleEndian.Uint32(data[record*32+16+row*4:]))
		}
	}
	for row := range rows {
		for column := 64; column < columns; column++ {
			want[row*columns+column] = 1
		}
		var total float32
		for _, value := range want[row*columns : (row+1)*columns] {
			total += value
		}
		inverse := 1 / total
		for i := row * columns; i < (row+1)*columns; i++ {
			want[i] *= inverse
		}
	}
	softmaxRows(values, rows, columns)
	for i, value := range values {
		if math.IsNaN(float64(want[i])) && math.IsNaN(float64(value)) {
			continue
		}
		if math.Float32bits(value) != math.Float32bits(want[i]) {
			t.Fatalf("output[%d]=%.9g (%08x), C oracle=%.9g (%08x)", i, value, math.Float32bits(value), want[i], math.Float32bits(want[i]))
		}
	}
}
