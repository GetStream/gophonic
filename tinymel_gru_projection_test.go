// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

func TestTinyGRUProjectionExact(t *testing.T) {
	rng := rand.New(rand.NewSource(8123))
	for _, dims := range [][3]int{{100, 192, 384}, {3, 2, 6}, {1, 192, 384}, {3, 17, 12}, {5, 127, 24}, {7, 128, 9}} {
		time, in, out := dims[0], dims[1], dims[2]
		input, weights := make([]float32, time*in), make([]float32, out*in)
		for i := range input {
			input[i] = float32(rng.NormFloat64())
		}
		for i := range weights {
			weights[i] = float32(rng.NormFloat64()) * 0.2
		}
		got, want := make([]float32, time*out), make([]float32, time*out)
		tinyGRUProjectInputsRowwise(input, weights, want, time, in, out)
		tinyGRUProjectInputs(input, weights, got, time, in, out)
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("dimensions=%v output[%d]=%.9g (%08x), want %.9g (%08x)", dims, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
		if allocs := testing.AllocsPerRun(3, func() { tinyGRUProjectInputs(input, weights, got, time, in, out) }); allocs != 0 {
			t.Fatalf("dimensions=%v allocated %g objects", dims, allocs)
		}
	}
}

func TestTinyGRUProjectionRealWeights(t *testing.T) {
	path := os.Getenv("GOPHONIC_TEST_TINYMEL_MODEL")
	if path == "" {
		t.Skip("set GOPHONIC_TEST_TINYMEL_MODEL for real projection parity")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		t.Fatal(err)
	}
	input := readFloatFixture(t, "testdata/tinymel_tone_gru_input.f32le")
	got, want := make([]float32, 100*384), make([]float32, 100*384)
	for direction := 0; direction < 2; direction++ {
		weights := model.gruW[direction*tinyGRUInputStride : (direction+1)*tinyGRUInputStride]
		tinyGRUProjectInputsRowwise(input, weights, want, 100, 192, 384)
		tinyGRUProjectInputs(input, weights, got, 100, 192, 384)
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("direction=%d projection[%d]=%.9g, want %.9g", direction, i, got[i], want[i])
			}
		}

	}
}

func BenchmarkTinyGRUProjection(b *testing.B) {
	input, weights, _, _ := tinyGRUTestTensors()
	output := make([]float32, 100*384)
	weights = weights[:tinyGRUInputStride]
	for _, impl := range []struct {
		name    string
		project func([]float32, []float32, []float32, int, int, int)
	}{
		{"rowwise", tinyGRUProjectInputsRowwise},
		{"tiled", tinyGRUProjectInputs},
	} {
		b.Run(impl.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				impl.project(input, weights, output, 100, 192, 384)
			}
		})
	}
}
