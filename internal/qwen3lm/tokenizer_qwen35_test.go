// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

// The Qwen3.5 family's tokenizer, whose pre-tokenizer counts combining
// marks as letters, matches Hugging Face tokenizers.
func TestQwen35Tokenizer(t *testing.T) {
	tok, err := LoadTokenizer(testmodels.Path(t, testmodels.Qwen36))
	if err != nil {
		t.Fatal(err)
	}
	if !tok.marks {
		t.Fatal("the Qwen3.5 expression was not recognized")
	}
	var ws TokenizerWorkspace
	for _, c := range []struct {
		text string
		ids  []int
	}{
		{"The capital of France is", []int{760, 6511, 314, 9338, 369}},
		{"नमस्ते दुनिया", []int{58069, 84237, 150104, 153348, 184642, 235886}},
		{"สวัสดีครับ", []int{34469, 168607, 153295}},
		{"Olá, você está bem? Ação!", []int{41304, 1886, 11, 23931, 15012, 30866, 30, 217838, 0}},
		{"é combining", []int{933, 33041}},
		{"Hello\n\n  world  <|im_end|>", []int{9419, 271, 220, 1814, 256, 248046}},
	} {
		got, err := tok.EncodeInto(c.text, make([]int, 0, 64), &ws)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, c.ids) {
			t.Errorf("%q: %v, want %v", c.text, got, c.ids)
		}
	}
}
