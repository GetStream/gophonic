// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor

import (
	"encoding/json"
	"math"
	"os"
	"runtime"
	"strconv"
	"testing"
)

func TestTinyQuantizeDynamicMatchesONNXRules(t *testing.T) {
	zeroInput := make([]float32, 8)
	zeroOutput := make([]uint8, len(zeroInput))
	params := quantizeTiny(zeroInput, zeroOutput)
	if params.scale != 1 || params.zero != 0 {
		t.Fatalf("zero-range quantization = (scale=%g,zp=%d), want (1,0)", params.scale, params.zero)
	}
	for i, value := range zeroOutput {
		if value != 0 {
			t.Fatalf("zero-range q[%d]=%d, want 0", i, value)
		}
	}

	input := []float32{-127.5, 127.5}
	output := make([]uint8, len(input))
	params = quantizeTiny(input, output)
	if params.scale != 1 || params.zero != 128 {
		t.Fatalf("half-tie quantization = (scale=%.9g,zp=%d), want (1,128)", params.scale, params.zero)
	}
	if output[0] != 0 || output[1] != 255 {
		t.Fatalf("half-tie q = %v, want [0 255]", output)
	}
}

func TestTinyMelOnnxOracleAndZeroAllocations(t *testing.T) {
	path := os.Getenv("GOFLOOR_TEST_TINYMEL_MODEL")
	if path == "" {
		t.Skip("set GOFLOOR_TEST_TINYMEL_MODEL to a converted TinyMelNet .gofloor bundle")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		t.Fatal(err)
	}
	ws := NewTinyMelWorkspace()
	defer ws.Close()

	zeroFeatures := make([]float32, tinyMelCount*tinyFrameCount)
	toneFeatures := readFloatFixture(t, "testdata/tone.mel.f32le")
	patternFeatures := readFloatFixture(t, "testdata/tinymel_pattern.mel.f32le")
	check := func(name string, features []float32, want float32) {
		t.Helper()
		got, err := model.PredictFeaturesInto(features, ws)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(got.Probability-want)) > 2e-5 {
			t.Errorf("%s probability = %.9f, want %.9f", name, got.Probability, want)
		}
	}
	check("all-zero features", zeroFeatures, 0.3462619483470917)
	check("440 Hz tone", toneFeatures, 0.016237027943134308)
	check("pattern", patternFeatures, 0.2457309365272522)
	checkTinyConvStages(t, model, toneFeatures, ws)

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := model.PredictFeaturesInto(toneFeatures, ws); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Errorf("steady-state TinyMel PredictFeaturesInto allocated %.1f objects per call, want zero", allocs)
	}

	pcm := readFloatFixture(t, "testdata/tone.pcm.f32le")
	endToEnd, err := model.PredictMono16kInto(pcm, ws)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(endToEnd.Probability-0.016237027943134308)) > 2e-5 {
		t.Errorf("440 Hz audio probability = %.9f, want %.9f", endToEnd.Probability, 0.016237027943134308)
	}
	endToEndAllocs := testing.AllocsPerRun(3, func() {
		if _, err := model.PredictMono16kInto(pcm, ws); err != nil {
			t.Fatal(err)
		}
	})
	if endToEndAllocs != 0 {
		t.Errorf("steady-state TinyMel PredictMono16kInto allocated %.1f objects per call, want zero", endToEndAllocs)
	}
}

func TestTinyMelWorkerWorkspaceParityAndZeroAllocations(t *testing.T) {
	path := os.Getenv("GOFLOOR_TEST_TINYMEL_MODEL")
	if path == "" {
		t.Skip("set GOFLOOR_TEST_TINYMEL_MODEL to a converted TinyMelNet .gofloor bundle")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name     string
		features []float32
	}{
		{name: "zero", features: make([]float32, tinyMelCount*tinyFrameCount)},
		{name: "tone", features: readFloatFixture(t, "testdata/tone.mel.f32le")},
		{name: "pattern", features: readFloatFixture(t, "testdata/tinymel_pattern.mel.f32le")},
	}
	previousProcs := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(previousProcs)

	serial := NewTinyMelWorkspace()
	wants := make([]Prediction, len(cases))
	for i, tc := range cases {
		wants[i], err = model.PredictFeaturesInto(tc.features, serial)
		if err != nil {
			serial.Close()
			t.Fatal(err)
		}
	}
	serial.Close()

	for helpers := 1; helpers <= MaxTinyMelWorkers; helpers++ {
		ws := NewTinyMelWorkspaceWithWorkers(helpers)
		if len(ws.workers) != helpers {
			t.Fatalf("requested %d helpers, got %d at GOMAXPROCS=8", helpers, len(ws.workers))
		}
		for i, tc := range cases {
			got, err := model.PredictFeaturesInto(tc.features, ws)
			if err != nil {
				ws.Close()
				t.Fatal(err)
			}
			if math.Abs(float64(got.Probability-wants[i].Probability)) > 2e-5 || got.Complete != wants[i].Complete {
				t.Errorf("%s helpers=%d prediction = %+v, serial = %+v", tc.name, helpers, got, wants[i])
			}
		}
		allocs := testing.AllocsPerRun(5, func() {
			if _, err := model.PredictFeaturesInto(cases[1].features, ws); err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Errorf("helpers=%d steady-state prediction allocated %.1f objects, want zero", helpers, allocs)
		}
		ws.Close()
	}
}

type tinyOracleFile struct {
	Cases []struct {
		Name    string `json:"name"`
		Tensors map[string]struct {
			Sample struct {
				FlatIndices []int     `json:"flatIndices"`
				Values      []float64 `json:"values"`
			} `json:"sample"`
		} `json:"tensors"`
		Quantization []struct {
			Node      string  `json:"node"`
			Scale     float64 `json:"scale"`
			ZeroPoint uint8   `json:"zeroPoint"`
		} `json:"quantization"`
	} `json:"cases"`
}

func checkTinyConvStages(t *testing.T, model *TinyMelModel, features []float32, ws *TinyMelWorkspace) {
	t.Helper()
	oracleBytes, err := os.ReadFile("testdata/tinymel_oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle tinyOracleFile
	if err := json.Unmarshal(oracleBytes, &oracle); err != nil {
		t.Fatal(err)
	}
	var tone *struct {
		Name    string `json:"name"`
		Tensors map[string]struct {
			Sample struct {
				FlatIndices []int     `json:"flatIndices"`
				Values      []float64 `json:"values"`
			} `json:"sample"`
		} `json:"tensors"`
		Quantization []struct {
			Node      string  `json:"node"`
			Scale     float64 `json:"scale"`
			ZeroPoint uint8   `json:"zeroPoint"`
		} `json:"quantization"`
	}
	for i := range oracle.Cases {
		if oracle.Cases[i].Name == "tone" {
			tone = &oracle.Cases[i]
			break
		}
	}
	if tone == nil {
		t.Fatal("TinyMel oracle is missing tone case")
	}
	checkStage := func(name string, values []float32, frames, channels int) {
		t.Helper()
		stage, ok := tone.Tensors[name]
		if !ok {
			t.Fatalf("TinyMel oracle is missing stage %q", name)
		}
		if len(stage.Sample.FlatIndices) != len(stage.Sample.Values) {
			t.Fatalf("TinyMel oracle stage %q has malformed samples", name)
		}
		for i, flat := range stage.Sample.FlatIndices {
			channel, frame := flat/frames, flat%frames
			if channel >= channels || frame*channels+channel >= len(values) {
				t.Fatalf("TinyMel oracle stage %q sample %d is outside output", name, flat)
			}
			got, want := values[frame*channels+channel], float32(stage.Sample.Values[i])
			if math.Abs(float64(got-want)) > 3e-5 {
				t.Errorf("TinyMel stage %s[%d] = %.8g, want %.8g", name, flat, got, want)
			}
		}
	}
	checkQuant := func(node string, params tinyQuantParams) {
		t.Helper()
		for _, want := range tone.Quantization {
			if want.Node == node {
				tolerance := math.Max(3e-9, math.Abs(want.Scale)*1e-6)
				if params.zero != want.ZeroPoint || math.Abs(float64(params.scale)-want.Scale) > tolerance {
					t.Errorf("TinyMel DQL %s = (%.9g,%d), want (%.9g,%d)", node, params.scale, params.zero, want.Scale, want.ZeroPoint)
				}
				return
			}
		}
		t.Fatalf("TinyMel oracle is missing DynamicQuantizeLinear node %q", node)
	}

	params := quantizeTinyMelForInference(features, ws.quantized, ws.melScratch)
	checkQuant("mel_QuantizeLinear", params)
	length := runTinyConv(model.stem, ws.quantized, params, ws.convA, tinyFrameCount)
	tinyGELUInPlace(ws.convA[:length*tinyStemChannels])
	checkStage("/stem/stem.2/Mul_1_output_0", ws.convA, length, tinyStemChannels)
	params = quantizeTiny(ws.convA[:length*tinyStemChannels], ws.quantized)
	checkQuant("/stem/stem.2/Mul_1_output_0_QuantizeLinear", params)

	depthNodes := [...]string{
		"/stem/stem.3/depthwise/Conv_output_0_QuantizeLinear",
		"/stem/stem.4/depthwise/Conv_output_0_QuantizeLinear",
		"/stem/stem.5/depthwise/Conv_output_0_QuantizeLinear",
	}
	depthStages := [...]string{
		"/stem/stem.3/depthwise/Conv_output_0",
		"/stem/stem.4/depthwise/Conv_output_0",
		"/stem/stem.5/depthwise/Conv_output_0",
	}
	pointStages := [...]string{
		"/stem/stem.3/pointwise/Conv_output_0",
		"/stem/stem.4/pointwise/Conv_output_0",
		"/stem/stem.5/pointwise/Conv_output_0",
	}
	actStages := [...]string{
		"/stem/stem.3/act/Mul_1_output_0",
		"/stem/stem.4/act/Mul_1_output_0",
		"/stem/stem.5/act/Mul_1_output_0",
	}
	actNodes := [...]string{
		"/stem/stem.3/act/Mul_1_output_0_QuantizeLinear",
		"/stem/stem.4/act/Mul_1_output_0_QuantizeLinear",
	}
	for i := range model.blocks {
		length = runTinyConv(model.blocks[i].depthwise, ws.quantized, params, ws.convB, length)
		checkStage(depthStages[i], ws.convB, length, tinyStemChannels)
		params = quantizeTiny(ws.convB[:length*tinyStemChannels], ws.quantized)
		checkQuant(depthNodes[i], params)
		length = runTinyConv(model.blocks[i].pointwise, ws.quantized, params, ws.convA, length)
		checkStage(pointStages[i], ws.convA, length, tinyStemChannels)
		tinyGELUInPlace(ws.convA[:length*tinyStemChannels])
		checkStage(actStages[i], ws.convA, length, tinyStemChannels)
		if i < len(actNodes) {
			params = quantizeTiny(ws.convA[:length*tinyStemChannels], ws.quantized)
			checkQuant(actNodes[i], params)
		}
	}
}

func BenchmarkTinyMelPredictFeatures(b *testing.B) {
	path := os.Getenv("GOFLOOR_TEST_TINYMEL_MODEL")
	if path == "" {
		b.Skip("set GOFLOOR_TEST_TINYMEL_MODEL to a converted TinyMelNet .gofloor bundle")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		b.Fatal(err)
	}
	features := readFloatFixture(b, "testdata/tone.mel.f32le")
	ws := NewTinyMelWorkspace()
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

func BenchmarkTinyMelPredictMono16k(b *testing.B) {
	path := os.Getenv("GOFLOOR_TEST_TINYMEL_MODEL")
	if path == "" {
		b.Skip("set GOFLOOR_TEST_TINYMEL_MODEL to a converted TinyMelNet .gofloor bundle")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		b.Fatal(err)
	}
	pcm := readFloatFixture(b, "testdata/tone.pcm.f32le")
	ws := NewTinyMelWorkspace()
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

func BenchmarkTinyMelWorkers(b *testing.B) {
	path := os.Getenv("GOFLOOR_TEST_TINYMEL_MODEL")
	if path == "" {
		b.Skip("set GOFLOOR_TEST_TINYMEL_MODEL to a converted TinyMelNet .gofloor bundle")
	}
	model, err := LoadTinyMel(path)
	if err != nil {
		b.Fatal(err)
	}
	features := readFloatFixture(b, "testdata/tone.mel.f32le")
	pcm := readFloatFixture(b, "testdata/tone.pcm.f32le")
	helperLimit := runtime.GOMAXPROCS(0) - 1
	if helperLimit > tinyMaxWorkers {
		helperLimit = tinyMaxWorkers
	}
	for helpers := 0; helpers <= helperLimit; helpers++ {
		for _, input := range []struct {
			name string
			run  func(*TinyMelWorkspace) (Prediction, error)
		}{
			{name: "features", run: func(ws *TinyMelWorkspace) (Prediction, error) { return model.PredictFeaturesInto(features, ws) }},
			{name: "audio", run: func(ws *TinyMelWorkspace) (Prediction, error) { return model.PredictMono16kInto(pcm, ws) }},
		} {
			name := input.name + "/helpers=" + strconv.Itoa(helpers)
			b.Run(name, func(b *testing.B) {
				ws := NewTinyMelWorkspaceWithWorkers(helpers)
				defer ws.Close()
				if _, err := input.run(ws); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := input.run(ws); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
