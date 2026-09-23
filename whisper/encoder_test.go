// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestConv1DChannelMajor(t *testing.T) {
	// One channel and three time steps exercise both zero-padded edges.
	src := []float32{10, 20, 30}
	weights := []float32{1, 2, 3}
	dst := make([]float32, 3)
	conv1DChannelMajor(src, dst, weights, []float32{0}, 3, 1, 1)
	want := []float32{80, 140, 80}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("output[%d] = %g, want %g", i, dst[i], want[i])
		}
	}
}

func TestConv1DStride2TimeMajor(t *testing.T) {
	src := []float32{10, 20, 30, 40}
	weights := []float32{1, 2, 3}
	dst := make([]float32, 2)
	conv1DStride2TimeMajor(src, dst, weights, []float32{0}, 4, 2, 1)
	want := []float32{80, 200}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("output[%d] = %g, want %g", i, dst[i], want[i])
		}
	}
}

func TestLayerNormRow(t *testing.T) {
	input := []float32{1, 2, 3}
	got := make([]float32, len(input))
	gamma := []float32{1, 2, 0.5}
	beta := []float32{0, 1, -2}
	layerNormRow(input, got, gamma, beta)

	invStd := 1 / math.Sqrt(2.0/3.0+1e-5)
	want := []float64{-invStd, 1, -2 + 0.5*invStd}
	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > 1e-6 {
			t.Fatalf("output[%d] = %.9g, want %.9g", i, got[i], want[i])
		}
	}
}

func TestExactGELUKnownValues(t *testing.T) {
	got := []float32{-1, 0, 1}
	applyGELU(got)
	want := []float32{-0.15865526, 0, 0.8413447}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-7 {
			t.Fatalf("GELU(%g) = %.9g, want %.9g", []float32{-1, 0, 1}[i], got[i], want[i])
		}
	}
}

func TestMultiHeadAttentionTwoTokenReference(t *testing.T) {
	// Identity Q/K and orthogonal V give a direct two-token softmax reference.
	q := []float32{1, 0, 0, 1}
	k := append([]float32(nil), q...)
	v := []float32{1, 0, 0, 1}
	scores := make([]float32, 2)
	multiHeadAttention(q, k, v, q, scores, 2, 2, 1)

	// Whisper scales Q and K by head_size^-1/4, which makes the nonzero
	// diagonal score 1/sqrt(2) before the softmax.
	diagonalScore := 1 / math.Sqrt(2)
	p0 := 1 / (1 + math.Exp(-diagonalScore))
	p1 := 1 - p0
	want := []float64{p0, p1, p1, p0}
	for i := range want {
		if math.Abs(float64(q[i])-want[i]) > 2e-7 {
			t.Fatalf("output[%d] = %.9g, want %.9g", i, q[i], want[i])
		}
	}
}

func TestEncoderPrimitiveKernelsDoNotAllocate(t *testing.T) {
	const n, d, heads = 2, 2, 1
	src := []float32{1, 2, 3, 4}
	dst := make([]float32, len(src))
	weights := []float32{1, 0, 0, 1}
	bias := []float32{0, 0}
	q := []float32{1, 0, 0, 1}
	k := []float32{1, 0, 0, 1}
	v := []float32{1, 0, 0, 1}
	scores := make([]float32, n)

	allocs := testing.AllocsPerRun(50, func() {
		linearRow(src[:d], dst[:d], weights, bias[:d])
		q[0], q[1], q[2], q[3] = 1, 0, 0, 1
		k[0], k[1], k[2], k[3] = 1, 0, 0, 1
		multiHeadAttention(q, k, v, q, scores, n, d, heads)
	})
	if allocs != 0 {
		t.Fatalf("warm encoder kernels allocated %.1f objects per run", allocs)
	}
}

func TestEncoderMatchesOfficialPyTorch(t *testing.T) {
	modelPath := os.Getenv("GOPHONIC_WHISPER_MODEL")
	if modelPath == "" {
		t.Skip("set GOPHONIC_WHISPER_MODEL to the converted official tiny.en bundle")
	}
	m, err := Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	fixtureDir := filepath.Join("..", "testdata", "whisper")
	mel, err := readFloatFixture(filepath.Join(fixtureDir, "jfk.mel.f32le"), MelBins*MelFrames)
	if err != nil {
		t.Fatal(err)
	}
	stages := []struct {
		name  string
		count int
	}{
		{"jfk.encoder.conv1.f32le", MelFrames * AudioState},
		{"jfk.encoder.conv2.f32le", AudioFrames * AudioState},
		{"jfk.encoder.block0.f32le", AudioFrames * AudioState},
		{"jfk.encoder.block1.f32le", AudioFrames * AudioState},
		{"jfk.encoder.block2.f32le", AudioFrames * AudioState},
		{"jfk.encoder.block3.f32le", AudioFrames * AudioState},
		{"jfk.encoder.f32le", AudioFrames * AudioState},
	}
	references := make([][]float32, len(stages))
	for i, stage := range stages {
		references[i], err = readFloatFixture(filepath.Join(fixtureDir, stage.name), stage.count)
		if err != nil {
			t.Fatal(err)
		}
	}

	var singleWorker []float32
	for _, workers := range []int{1, 8} {
		t.Run(fmt.Sprintf("workers%d", workers), func(t *testing.T) {
			workspace, err := NewEncoderWorkspaceWithWorkers(workers)
			if err != nil {
				t.Fatal(err)
			}
			defer workspace.Close()
			got := make([]float32, AudioFrames*AudioState)
			nextStage := 0
			var stageErr error
			err = m.encodeInto(mel, got, workspace, func(stageIndex int, values []float32) {
				if stageErr != nil {
					return
				}
				if stageIndex != nextStage || nextStage >= len(stages) {
					stageErr = fmt.Errorf("encoder emitted unexpected stage %d, want %d", stageIndex, nextStage)
					return
				}
				stage := stages[nextStage]
				stageErr = compareEncoderStage(stage.name, values, references[nextStage])
				nextStage++
			})
			if err != nil {
				t.Fatal(err)
			}
			if stageErr != nil {
				t.Fatal(stageErr)
			}
			if nextStage != len(stages) {
				t.Fatalf("encoder emitted %d stages; want %d", nextStage, len(stages))
			}
			if workers == 1 {
				singleWorker = append([]float32(nil), got...)
			} else {
				if !slices.Equal(got, singleWorker) {
					t.Fatal("worker count changed encoder results")
				}
				if err := m.EncodeInto(mel, got, workspace); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, singleWorker) {
					t.Fatal("reusing the workspace changed encoder results")
				}
			}
		})
	}
}

func TestEncoderWarmCallDoesNotAllocate(t *testing.T) {
	if os.Getenv("GOPHONIC_TEST_ENCODER_ALLOCS") != "1" {
		t.Skip("set GOPHONIC_TEST_ENCODER_ALLOCS=1 to run two full encoder calls")
	}
	modelPath := os.Getenv("GOPHONIC_WHISPER_MODEL")
	if modelPath == "" {
		t.Skip("set GOPHONIC_WHISPER_MODEL to the converted official tiny.en bundle")
	}
	m, err := Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	mel, err := readFloatFixture(filepath.Join("..", "testdata", "whisper", "jfk.mel.f32le"), MelBins*MelFrames)
	if err != nil {
		t.Fatal(err)
	}
	w := NewEncoderWorkspace()
	defer w.Close()
	dst := make([]float32, AudioFrames*AudioState)
	allocs := testing.AllocsPerRun(1, func() {
		if err := m.EncodeInto(mel, dst, w); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm EncodeInto allocated %.1f objects per call", allocs)
	}
}

func readFloatFixture(path string, count int) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) != count*4 {
		return nil, fmt.Errorf("%s contains %d bytes; want %d", path, len(data), count*4)
	}
	values := make([]float32, count)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return values, nil
}

func compareEncoderStage(name string, got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s has %d values; want %d", name, len(got), len(want))
	}
	const absoluteTolerance = 2e-4
	const relativeTolerance = 2e-4
	for i, actual := range got {
		expected := want[i]
		if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) ||
			math.IsNaN(float64(expected)) || math.IsInf(float64(expected), 0) {
			return fmt.Errorf("%s[%d] contains non-finite actual %g or expected %g", name, i, actual, expected)
		}
		error := math.Abs(float64(actual - expected))
		limit := absoluteTolerance + relativeTolerance*math.Abs(float64(expected))
		if error > limit {
			return fmt.Errorf("%s[%d] = %.8g, want %.8g (abs error %.4g exceeds %.4g)", name, i, actual, expected, error, limit)
		}
	}
	return nil
}

func TestEncoderWorkspaceClose(t *testing.T) {
	w := NewEncoderWorkspace()
	if !w.valid() {
		t.Fatal("new workspace is missing required scratch")
	}
	w.Close()
	if w.valid() || !w.closed {
		t.Fatal("closed workspace remains usable")
	}
	if err := (*Model)(nil).EncodeInto(nil, nil, w); err != errNilModel {
		t.Fatalf("nil model error = %v, want %v", err, errNilModel)
	}
}
