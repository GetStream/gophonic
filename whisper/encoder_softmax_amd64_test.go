// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whisper

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSIMDEncoderPreservesOfficialGreedyTokens(t *testing.T) {
	modelPath := os.Getenv("GOPHONIC_WHISPER_MODEL")
	if modelPath == "" {
		t.Skip("set GOPHONIC_WHISPER_MODEL to the converted official tiny.en bundle")
	}
	m, err := Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join("..", "testdata", "whisper")
	mel, err := readFloatFixture(filepath.Join(dir, "jfk.mel.f32le"), MelBins*MelFrames)
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := os.ReadFile(filepath.Join(dir, "jfk.oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var oracle decoderOracle
	if err := json.Unmarshal(manifest, &oracle); err != nil {
		t.Fatal(err)
	}
	workspace := NewEncoderWorkspace()
	defer workspace.Close()
	encoded := make([]float32, AudioFrames*AudioState)
	if err := m.EncodeInto(mel, encoded, workspace); err != nil {
		t.Fatal(err)
	}
	decoder := NewDecoderScratch()
	decoder.gemm = workspace.gemm
	output := make([]int, len(oracle.Prefix)+len(oracle.Tokens)+1)
	n, err := m.GreedyDecodeInto(encoded, oracle.Prefix, output, decoder, 50256)
	if err != nil || n != len(output) {
		t.Fatalf("greedy count=%d, want %d; error=%v", n, len(output), err)
	}
	for i, tokenID := range oracle.Prefix {
		if output[i] != tokenID {
			t.Fatalf("prefix[%d]=%d, want %d", i, output[i], tokenID)
		}
	}
	for i, tokenID := range oracle.Tokens {
		if output[len(oracle.Prefix)+i] != tokenID {
			t.Fatalf("generated token[%d]=%d, want %d", i, output[len(oracle.Prefix)+i], tokenID)
		}
	}
	if output[n-1] != 50256 {
		t.Fatalf("final token=%d, want EOT", output[n-1])
	}
}

func TestSoftmaxExpMatchesPinnedNEON(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "softmax_exp_neon.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 64*32 {
		t.Fatalf("NEON fixture size=%d, want %d", len(data), 64*32)
	}
	if digest := fmt.Sprintf("%x", sha256.Sum256(data)); digest != "d80ca9601cfe78afc049731354eb4b5619c83858429ac6085039a45a9b665423" {
		t.Fatalf("unexpected pinned NEON fixture digest: %s", digest)
	}
	for record := 0; record < len(data); record += 32 {
		var got [4]float32
		for lane := range got {
			got[lane] = math.Float32frombits(binary.LittleEndian.Uint32(data[record+lane*4:]))
		}
		softmaxExpInPlace(got[:], 0)
		for lane, value := range got {
			wantBits := binary.LittleEndian.Uint32(data[record+16+lane*4:])
			want := math.Float32frombits(wantBits)
			if math.IsNaN(float64(want)) {
				if !math.IsNaN(float64(value)) {
					t.Fatalf("record=%d lane=%d output=%g, want NaN", record/32, lane, value)
				}
			} else if math.Float32bits(value) != wantBits {
				t.Fatalf("record=%d lane=%d output=%.9g (%08x), C=%.9g (%08x)", record/32, lane, value, math.Float32bits(value), want, wantBits)
			}
		}
	}
}

func TestSoftmaxExpNonpositiveAccuracy(t *testing.T) {
	// Include a dense grid across the normal/subnormal boundary and values
	// adjacent to argument-reduction boundaries, not only model-like logits.
	inputs := make([]float32, 0, 65536+456)
	for i := range 65536 {
		inputs = append(inputs, -float32(i)*110/65535)
	}
	for exponent := -150; exponent <= 0; exponent++ {
		value := float32(float64(exponent) * math.Ln2)
		inputs = append(inputs, math.Nextafter32(value, float32(math.Inf(-1))), value, min(0, math.Nextafter32(value, 0)))
	}
	values := append([]float32(nil), inputs...)
	softmaxExpInPlace(values, 0)
	var maxULP uint32
	for i, input := range inputs {
		want := float32(math.Exp(float64(input)))
		gotBits, wantBits := math.Float32bits(values[i]), math.Float32bits(want)
		difference := max(gotBits, wantBits) - min(gotBits, wantBits)
		maxULP = max(maxULP, difference)
		if math.IsNaN(float64(values[i])) || difference > 2 {
			t.Fatalf("exp(%g)=%.9g, want %.9g: %d ulps", input, values[i], want, difference)
		}
	}
	t.Logf("maximum observed exponential error: %d ulps", maxULP)
}

func TestSoftmaxSIMDTailsAndSpecialValues(t *testing.T) {
	for _, length := range []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 65, 1500} {
		for offset := range 4 {
			storage := make([]float32, length+offset)
			values := storage[offset:]
			for i := range values {
				values[i] = float32(30*math.Sin(float64(i)*0.31) - 40)
			}
			want := append([]float32(nil), values...)
			referenceSoftmax(want)
			softmaxRow(values)
			for i, value := range values {
				if math.IsNaN(float64(value)) || math.Abs(float64(value-want[i])) > 2e-7 {
					t.Fatalf("length=%d offset=%d output[%d]=%.9g, want %.9g", length, offset, i, value, want[i])
				}
			}
		}
	}
	for _, input := range [][]float32{
		{0, float32(math.Inf(-1)), -1, -100, -104},
		{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))},
		{1, float32(math.Inf(1)), 2, 3},
		{1, 2, float32(math.NaN()), 4, 5},
	} {
		got, want := append([]float32(nil), input...), append([]float32(nil), input...)
		softmaxRow(got)
		referenceSoftmax(want)
		for i, value := range got {
			if math.IsNaN(float64(want[i])) {
				if !math.IsNaN(float64(value)) {
					t.Fatalf("input=%v output[%d]=%g, want NaN", input, i, value)
				}
			} else if math.IsNaN(float64(value)) || math.Abs(float64(value-want[i])) > 2e-7 {
				t.Fatalf("input=%v output[%d]=%g, want %g", input, i, value, want[i])
			}
		}
	}
}

func referenceSoftmax(values []float32) {
	maximum := values[0]
	for _, value := range values[1:] {
		maximum = max(maximum, value)
	}
	var total float32
	for i, value := range values {
		values[i] = float32(math.Exp(float64(value - maximum)))
		total += values[i]
	}
	inverse := 1 / total
	for i := range values {
		values[i] *= inverse
	}
}
