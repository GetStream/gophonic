// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"os"
	"slices"
	"testing"
)

// These IDs were generated with AutoTokenizer from Qwen/Qwen3-8B revision
// b968826d9c46dd6066d109eabc6255188de91218 using add_special_tokens=False.
// Provide the official snapshot path to run this gate; CI without weights skips.
func TestOfficialQwen3TokenizerParity(t *testing.T) {
	path := os.Getenv("GOPHONIC_QWEN3_TOKENIZER")
	if path == "" {
		t.Skip("set GOPHONIC_QWEN3_TOKENIZER to the official Qwen3-8B snapshot")
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	var ws TokenizerWorkspace
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
		got, err := tok.EncodeInto(tc.input, make([]int, 0, len(tc.input)+8), &ws)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, tc.want) {
			t.Errorf("%q: got %v, want %v", tc.input, got, tc.want)
		}
	}
}
