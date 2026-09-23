// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"math"
	"os"
	"runtime"
	"strconv"
	"testing"
)

func BenchmarkTinyMelStageBreakdown(b *testing.B) {
	path := os.Getenv("GOPHONIC_TEST_TINYMEL_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_TEST_TINYMEL_MODEL to a converted TinyMelNet .gophonic bundle")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		b.Fatal(err)
	}
	features := readFloatFixture(b, "testdata/tone.mel.f32le")
	helperCount := runtime.GOMAXPROCS(0) - 1
	if helperCount > tinyMaxWorkers {
		helperCount = tinyMaxWorkers
	}
	ws := NewTinyMelWorkspaceWithWorkers(helperCount)
	defer ws.Close()
	b.ReportMetric(float64(len(ws.workers)+1), "lanes")

	b.Run("stem", func(b *testing.B) {
		model.encodeTinyStem(features, ws)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			model.encodeTinyStem(features, ws)
		}
	})

	inputLengths := [...]int{tinyFrameCount / 2, tinyFrameCount / 4, tinySequenceLength}
	for block := range model.blocks {
		// Rebuild the preceding activations outside the timed region so each
		// block starts with the correct shape and representative model values.
		length := model.encodeTinyStem(features, ws)
		for previous := 0; previous < block; previous++ {
			length = model.encodeTinyBlock(previous, length, ws)
		}
		if length != inputLengths[block] {
			b.Fatalf("block %d input length=%d, want %d", block, length, inputLengths[block])
		}
		b.Run("block"+strconv.Itoa(block), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				model.encodeTinyBlock(block, inputLengths[block], ws)
			}
		})
	}

	// Restore the complete encoder activation before timing the recurrent stage.
	length := model.encodeTinyStem(features, ws)
	for block := range model.blocks {
		length = model.encodeTinyBlock(block, length, ws)
	}
	gruInput := ws.convA[:length*tinyStemChannels]
	b.Run("gru", func(b *testing.B) {
		ws.runGRU(model, gruInput)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ws.runGRU(model, gruInput)
		}
	})

	ws.runGRU(model, gruInput)
	b.Run("attention+head", func(b *testing.B) {
		runTinyAttentionHead(model, ws)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			runTinyAttentionHead(model, ws)
		}
	})
}

func runTinyAttentionHead(model *TinyMelModel, ws *TinyMelWorkspace) {
	tinyAttentionPool(model, ws)
	tinyLayerNorm(ws.pooled, ws.headNormed, model.headNormW, model.headNormB)
	params := quantizeTiny(ws.headNormed, ws.quantized)
	tinyQuantizedMatMul(model.head1, ws.quantized, params, ws.headHidden)
	tinyGELUInPlace(ws.headHidden)
	params = quantizeTiny(ws.headHidden, ws.quantized)
	tinyQuantizedMatMul(model.head2, ws.quantized, params, ws.logit[:])
}

func BenchmarkTinyMelEncoderSubstages(b *testing.B) {
	path := os.Getenv("GOPHONIC_TEST_TINYMEL_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_TEST_TINYMEL_MODEL to a converted TinyMelNet .gophonic bundle")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		b.Fatal(err)
	}
	features := readFloatFixture(b, "testdata/tone.mel.f32le")
	helperCount := runtime.GOMAXPROCS(0) - 1
	if helperCount > tinyMaxWorkers {
		helperCount = tinyMaxWorkers
	}
	ws := NewTinyMelWorkspaceWithWorkers(helperCount)
	defer ws.Close()

	melParams := quantizeTinyMel(features, ws.quantized)
	stemLength := ws.runConv(model.stem, ws.quantized, melParams, ws.convA, tinyFrameCount)
	if stemLength != tinyFrameCount/2 {
		b.Fatalf("stem length=%d", stemLength)
	}
	stemValues := append([]float32(nil), ws.convA[:stemLength*tinyStemChannels]...)
	stemGELUOut := make([]float32, len(stemValues))
	blockLength := model.encodeTinyBlock(0, stemLength, ws)
	block0Values := append([]float32(nil), ws.convA[:blockLength*tinyStemChannels]...)
	block0GELUOut := make([]float32, len(block0Values))

	b.Run("stem-mel-quantize", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			quantizeTinyMelForInference(features, ws.quantized, ws.melScratch)
		}
	})
	b.Run("stem-conv", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ws.runConv(model.stem, ws.quantized, melParams, ws.convA, tinyFrameCount)
		}
	})
	benchGELU := func(name string, input, output []float32) {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				for i, value := range input {
					output[i] = 0.5 * value * float32(1+math.Erf(float64(value*0.7071067811865475244)))
				}
			}
		})
	}
	benchGELU("stem-gelu", stemValues, stemGELUOut)
	benchGELU("block0-gelu", block0Values, block0GELUOut)
}
