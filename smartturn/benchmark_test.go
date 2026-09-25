// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package smartturn

import (
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

func BenchmarkFeatureExtraction16k(b *testing.B) {
	pcm := readFloatFixture(b, "../testdata/tone.pcm.f32le")
	features := make([]float32, melCount*frameCount)
	ws := NewWorkspace()
	defer ws.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := FeaturesInto(pcm, features, ws); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPredictFeatures(b *testing.B) {
	model, features := benchmarkModelAndFeatures(b)
	ws := NewWorkspace()
	defer ws.Close()
	if _, err := model.PredictFeaturesInto(features, ws); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.PredictFeaturesInto(features, ws); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPredictMono16k(b *testing.B) {
	model, _ := benchmarkModelAndFeatures(b)
	pcm := readFloatFixture(b, "../testdata/tone.pcm.f32le")
	ws := NewWorkspace()
	defer ws.Close()
	if _, err := model.PredictMono16kInto(pcm, ws); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.PredictMono16kInto(pcm, ws); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkModelAndFeatures(b *testing.B) (*Model, []float32) {
	b.Helper()
	path := testmodels.Path(b, testmodels.SmartTurn)
	model, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	return model, readFloatFixture(b, "../testdata/tone.mel.f32le")
}
