// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

type decoderOracle struct {
	Prefix       []int  `json:"prefix"`
	PrefixArgmax int    `json:"prefix_argmax"`
	Tokens       []int  `json:"tokens"`
	Transcript   string `json:"transcript"`
}

func TestDecoderMathHelpers(t *testing.T) {
	t.Run("linear row-major", func(t *testing.T) {
		// Two outputs, three inputs, with a separate output bias.
		weight := []float32{1, 2, 3, -1, 0.5, 4}
		bias := []float32{0.25, -0.5}
		x := []float32{2, -1, 0.5}
		got := make([]float32, 2)
		linearInto(got, x, weight, bias, 3, 2)
		if got[0] != 1.75 || got[1] != -1 {
			t.Fatalf("linear output %v, want [1.75 -1]", got)
		}
	})

	t.Run("layer norm population variance", func(t *testing.T) {
		x := []float32{1, 2, 3, 4}
		weight := []float32{1, 1, 1, 1}
		bias := []float32{0, 0, 0, 0}
		got := make([]float32, 4)
		layerNorm(got, x, weight, bias)
		want := []float32{-1.3416355, -0.44721183, 0.44721183, 1.3416355}
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 1e-6 {
				t.Fatalf("layer norm[%d]=%.8g, want %.8g", i, got[i], want[i])
			}
		}
	})

	t.Run("single-frame multihead attention", func(t *testing.T) {
		// A one-frame softmax has probability exactly one, so the result must be
		// the value vector regardless of the query/key scale.
		query := []float32{1, 2, 3, 4}
		keys := []float32{2, 1, 0, -1}
		values := []float32{5, 6, 7, 8}
		got := make([]float32, 4)
		attentionInto(got, query, keys, values, 1, 2, make([]float32, 1))
		for i := range got {
			if got[i] != values[i] {
				t.Fatalf("attention[%d]=%g, want %g", i, got[i], values[i])
			}
		}
	})

	t.Run("exact GELU and first argmax tie", func(t *testing.T) {
		values := []float32{-1, 0, 1}
		geluExactInto(values)
		want := []float32{-0.15865526, 0, 0.8413447}
		for i := range values {
			if math.Abs(float64(values[i]-want[i])) > 1e-6 {
				t.Fatalf("GELU[%d]=%.8g, want %.8g", i, values[i], want[i])
			}
		}
		if got := argmax([]float32{1, 2, 2, -1}); got != 1 {
			t.Fatalf("argmax tie selected %d, want first index 1", got)
		}
	})
}

func TestDecoderOfficialPrefixAndGreedyTokenOracle(t *testing.T) {
	modelPath := os.Getenv("GOPHONIC_WHISPER_MODEL")
	if modelPath == "" {
		t.Skip("set GOPHONIC_WHISPER_MODEL to the converted official tiny.en weights")
	}
	m, err := Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}

	fixtureDir := filepath.Join("..", "testdata", "whisper")
	var oracle decoderOracle
	manifest, err := os.ReadFile(filepath.Join(fixtureDir, "jfk.oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(manifest, &oracle); err != nil {
		t.Fatal(err)
	}
	encoder, err := readF32Fixture(filepath.Join(fixtureDir, "jfk.encoder.f32le"), AudioFrames*AudioState)
	if err != nil {
		t.Fatal(err)
	}
	wantLogits, err := readF32Fixture(filepath.Join(fixtureDir, "jfk.prefix_logits.f32le"), VocabSize)
	if err != nil {
		t.Fatal(err)
	}
	if len(oracle.Prefix) == 0 || len(oracle.Tokens) == 0 {
		t.Fatal("oracle manifest is missing the prefix or decoded tokens")
	}

	s := NewDecoderScratch()
	if err := m.BeginDecode(encoder, s); err != nil {
		t.Fatal(err)
	}
	for position, tokenID := range oracle.Prefix {
		if err := m.LogitsForTokenInto(tokenID, position, s, s.logits); err != nil {
			t.Fatal(err)
		}
	}
	if got := argmax(s.logits); got != oracle.PrefixArgmax {
		t.Fatalf("prefix argmax %d, want %d", got, oracle.PrefixArgmax)
	}
	var maxAbs, sumSq float64
	for i, got := range s.logits {
		diff := math.Abs(float64(got - wantLogits[i]))
		if diff > maxAbs {
			maxAbs = diff
		}
		sumSq += diff * diff
	}
	rms := math.Sqrt(sumSq / float64(VocabSize))
	if maxAbs > 2e-2 || rms > 2e-3 {
		t.Fatalf("prefix logits differ from official PyTorch: max abs %.6g, RMS %.6g", maxAbs, rms)
	}
	// The next position must consume the self-KV cached by the full prefix.
	afterFirst, err := readF32Fixture(filepath.Join(fixtureDir, "jfk.after_843_logits.f32le"), VocabSize)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.LogitsForTokenInto(843, len(oracle.Prefix), s, s.logits); err != nil {
		t.Fatal(err)
	}
	compareDecoderLogits(t, "after first generated token", s.logits, afterFirst)

	// A fresh audio window must replace both cross-KV and self-KV state.
	halfAudio := make([]float32, len(encoder))
	for i, value := range encoder {
		halfAudio[i] = value * 0.5
	}
	halfWant, err := readF32Fixture(filepath.Join(fixtureDir, "jfk.half_audio_prefix_logits.f32le"), VocabSize)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.BeginDecode(halfAudio, s); err != nil {
		t.Fatal(err)
	}
	for position, tokenID := range oracle.Prefix {
		if err := m.LogitsForTokenInto(tokenID, position, s, s.logits); err != nil {
			t.Fatal(err)
		}
	}
	compareDecoderLogits(t, "second audio window", s.logits, halfWant)

	output := make([]int, len(oracle.Prefix)+len(oracle.Tokens)+1)
	n, err := m.GreedyDecodeInto(encoder, oracle.Prefix, output, s, 50256)
	if err != nil {
		t.Fatal(err)
	}
	wantN := len(oracle.Prefix) + len(oracle.Tokens) + 1 // Include the selected EOT.
	if n != wantN {
		t.Fatalf("greedy output has %d IDs, want %d (including EOT)", n, wantN)
	}
	for i, tokenID := range oracle.Prefix {
		if output[i] != tokenID {
			t.Fatalf("output prefix[%d]=%d, want %d", i, output[i], tokenID)
		}
	}
	for i, tokenID := range oracle.Tokens {
		if got := output[len(oracle.Prefix)+i]; got != tokenID {
			t.Fatalf("greedy token[%d]=%d, want official token %d", i, got, tokenID)
		}
	}
	if got := output[n-1]; got != 50256 {
		t.Fatalf("greedy final token=%d, want EOT 50256", got)
	}

	// Measure the complete steady-state token operation, including a fresh
	// cross-attention cache for the next input. Weight packing is already warm.
	if err := m.BeginDecode(encoder, s); err != nil {
		t.Fatal(err)
	}
	if err := m.LogitsForTokenInto(oracle.Prefix[0], 0, s, s.logits); err != nil {
		t.Fatal(err)
	}
	var hotErr error
	allocs := testing.AllocsPerRun(3, func() {
		hotErr = m.BeginDecode(encoder, s)
		if hotErr == nil {
			hotErr = m.LogitsForTokenInto(oracle.Prefix[0], 0, s, s.logits)
		}
	})
	if hotErr != nil {
		t.Fatal(hotErr)
	}
	if allocs != 0 {
		t.Fatalf("warm BeginDecode + token step allocated %.2f objects", allocs)
	}
	for _, workers := range []int{1, 8} {
		pool, err := whispergemm.NewExecutor(workers)
		if err != nil {
			t.Fatal(err)
		}
		s.gemm = pool
		if err := m.BeginDecode(encoder, s); err != nil {
			t.Fatal(err)
		}
		for position, tokenID := range oracle.Prefix {
			if err := m.LogitsForTokenInto(tokenID, position, s, s.logits); err != nil {
				t.Fatal(err)
			}
		}
		compareDecoderLogits(t, "borrowed executor prefix", s.logits, wantLogits)
		n, err := m.GreedyDecodeInto(encoder, oracle.Prefix, output, s, 50256)
		if err != nil || n != wantN || output[n-1] != 50256 {
			t.Fatalf("workers=%d: count %d error %v", workers, n, err)
		}
		for i, want := range oracle.Tokens {
			if got := output[len(oracle.Prefix)+i]; got != want {
				t.Fatalf("workers=%d token[%d]=%d want %d", workers, i, got, want)
			}
		}
		allocs := testing.AllocsPerRun(2, func() {
			hotErr = m.BeginDecode(encoder, s)
			if hotErr == nil {
				hotErr = m.LogitsForTokenInto(oracle.Prefix[0], 0, s, s.logits)
			}
		})
		if hotErr != nil || allocs != 0 {
			t.Fatalf("workers=%d warm BeginDecode + token: allocs=%g error=%v", workers, allocs, hotErr)
		}
		pool.Close()
		s.gemm = nil
	}
}

func compareDecoderLogits(t *testing.T, stage string, got, want []float32) {
	t.Helper()
	var maxAbs, sumSq float64
	for i, value := range got {
		diff := math.Abs(float64(value - want[i]))
		if diff > maxAbs {
			maxAbs = diff
		}
		sumSq += diff * diff
	}
	rms := math.Sqrt(sumSq / float64(len(got)))
	if maxAbs > 2e-2 || rms > 2e-3 {
		t.Fatalf("%s logits differ from official PyTorch: max abs %.6g, RMS %.6g", stage, maxAbs, rms)
	}
}

func readF32Fixture(path string, count int) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) != count*4 {
		return nil, os.ErrInvalid
	}
	values := make([]float32, count)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4 : i*4+4]))
	}
	return values, nil
}
