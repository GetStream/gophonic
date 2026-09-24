// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"context"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/clm"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestCopyHidden(t *testing.T) {
	input := make([]float32, hiddenSize)
	input[0], input[1] = 3, 4
	out := make([]float32, hiddenSize)
	if err := copyHidden(out, input); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(out[0]-3)) > 1e-6 || math.Abs(float64(out[1]-4)) > 1e-6 {
		t.Fatalf("got first values %.8f, %.8f; want 3, 4", out[0], out[1])
	}
	input[0] = float32(math.Inf(1))
	if err := copyHidden(out, input); err == nil {
		t.Fatal("accepted a non-finite embedding")
	}
}

// These IDs were generated with AutoTokenizer from Qwen/Qwen3-8B revision
// b968826d9c46dd6066d109eabc6255188de91218 using add_special_tokens=False.
// Provide the official snapshot path to run this gate; CI without weights skips.
func TestOfficialQwen3TokenizerParity(t *testing.T) {
	path := os.Getenv("GOPHONIC_QWEN3_TOKENIZER")
	if path == "" {
		t.Skip("set GOPHONIC_QWEN3_TOKENIZER to the official Qwen3-8B snapshot")
	}
	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		input string
		want  []int
	}{
		{"hello", []int{14990}},
		{"What causes tides on Earth?", []int{3838, 11137, 259, 3341, 389, 9237, 30}},
		{"Customer: my invoice was charged twice and nobody answers the phone!", []int{12792, 25, 847, 24615, 572, 11430, 10917, 323, 18581, 11253, 279, 4540, 0}},
		{"tool: search(query=\"Go\")", []int{14172, 25, 2711, 10741, 428, 10850, 899}},
	}
	for _, tc := range cases {
		got, err := tok.Encode(tc.input, false)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, tc.want) {
			t.Errorf("%q: got %v, want %v", tc.input, got, tc.want)
		}
	}
}

// TestOfficialCLMRanking is an opt-in full-model correctness gate. The golden
// probabilities come from official Qwen3-8B BF16 hidden states followed by the
// published CLM v0.1 PyTorch state/action heads on this exact candidate set.
func TestOfficialCLMRanking(t *testing.T) {
	qwenPath := os.Getenv("GOPHONIC_QWEN3_MODEL")
	headPath := os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE")
	if qwenPath == "" || headPath == "" {
		t.Skip("set GOPHONIC_QWEN3_MODEL and GOPHONIC_CLM_HEAD_BUNDLE for full-model parity")
	}
	head, err := clm.Load(headPath)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := Open(qwenPath)
	if err != nil {
		t.Fatal(err)
	}
	defer encoder.Close()
	engine, err := clm.NewEngine(head, encoder)
	if err != nil {
		t.Fatal(err)
	}
	result, err := engine.Rank(context.Background(), "What causes tides on Earth?", []string{
		"The Moon’s gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round.",
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	wantNames := []string{"The Moon’s gravitational pull.", "Because the Earth is round.", "Photosynthesis in plants."}
	wantProb := []float64{0.9981802701950073, 0.0018110087839886546, 8.711908776604105e-06}
	for i, got := range result {
		if got.Candidate != wantNames[i] || got.Rank != i+1 {
			t.Errorf("rank %d: got %+v, want %q", i+1, got, wantNames[i])
		}
		if math.Abs(float64(got.Probability)-wantProb[i]) > 0.001 {
			t.Errorf("rank %d probability %.8f, BF16 reference %.8f", i+1, got.Probability, wantProb[i])
		}
	}
}
