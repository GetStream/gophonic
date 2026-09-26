// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/thesyncim/vibejson"
)

func TestQwenTokenizerHFTokenIDGoldens(t *testing.T) {
	tk, err := LoadTokenizer(qwenTokenizerDir(t))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		text string
		ids  []int
	}{
		{"hello", []int{14990}},
		{"we're READY, I'M here!", []int{896, 2299, 85332, 11, 358, 27603, 1588, 0}},
		{"e\u0301 cafe\u0301 — 你好，世界🙂", []int{963, 51950, 1959, 220, 108386, 3837, 99489, 145080}},
		{"ASCII\tspace  \r\n line\n12345 ١٢٣ Ⅷ", []int{56450, 1903, 1306, 10419, 1555, 198, 16, 17, 18, 19, 20, 220, 149, 94, 149, 95, 149, 96, 220, 70467, 100}},
		{" punctuation!!!\n\nNext\rEND", []int{61503, 32057, 5847, 201, 4689}},
		{"A<|im_start|>user\nHi<|im_end|>", []int{32, 151644, 872, 198, 13048, 151645}},
		{"<|vision_start|>hello<|vision_end|> <|audio|>", []int{151652, 14990, 151653, 82639, 16736, 91, 29}},
		{"Qwen3 CLM state/action ranking: 42 candidates.", []int{48, 16948, 18, 6976, 44, 1584, 54765, 23001, 25, 220, 19, 17, 11178, 13}},
	}
	var ws TokenizerWorkspace
	for _, tc := range cases {
		t.Run(tc.text, func(t *testing.T) {
			got, err := tk.EncodeInto(tc.text, make([]int, 0, len(tc.text)*4+32), &ws)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.ids) {
				t.Fatalf("EncodeInto(%q) = %v, want HF ids %v", tc.text, got, tc.ids)
			}
		})
	}
}

// TestQwenTokenizerOfficialGoldens compares Unicode, whitespace, special-token,
// and long inputs with IDs produced by the official Hugging Face tokenizer
// (testdata/qwen3_tokenizer_goldens.json records the exact source version).
func TestQwenTokenizerOfficialGoldens(t *testing.T) {
	dir := qwenTokenizerDir(t)
	ours, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/qwen3_tokenizer_goldens.json")
	if err != nil {
		t.Fatal(err)
	}
	var goldens struct {
		Cases []struct {
			Text string `json:"text"`
			IDs  []int  `json:"ids"`
		} `json:"cases"`
	}
	if err := vibejson.Unmarshal(raw, &goldens); err != nil {
		t.Fatal(err)
	}
	if len(goldens.Cases) == 0 {
		t.Fatal("no golden cases")
	}
	var ws TokenizerWorkspace
	for _, tc := range goldens.Cases {
		got, err := ours.EncodeInto(tc.Text, make([]int, 0, len(tc.Text)*4+32), &ws)
		if err != nil {
			t.Fatalf("EncodeInto(%q): %v", shortText(tc.Text), err)
		}
		if len(got) != len(tc.IDs) || (len(got) != 0 && !reflect.DeepEqual(got, tc.IDs)) {
			t.Fatalf("token ids differ for %q: got %v, want %v", shortText(tc.Text), got[:min(len(got), 24)], tc.IDs[:min(len(tc.IDs), 24)])
		}
	}
}

func TestQwenTokenizerOfficialZeroAllocAfterWarmup(t *testing.T) {
	tk, err := LoadTokenizer(qwenTokenizerDir(t))
	if err != nil {
		t.Fatal(err)
	}
	inputs := []string{
		"hello",
		"The Moon's gravitational pull explains Earth's tides.",
		"e\u0301 cafe\u0301 — 你好，世界🙂 <|im_start|>user\nHi<|im_end|>",
		strings.Repeat("Qwen3 state/action ranking 2026!\r\n", 128),
	}
	dst := make([]int, 0, 4*len(inputs[len(inputs)-1])+32)
	var ws TokenizerWorkspace
	for _, input := range inputs {
		out, err := tk.EncodeInto(input, dst[:0], &ws)
		if err != nil {
			t.Fatalf("warm %q: %v", shortText(input), err)
		}
		dst = out
	}
	for _, input := range inputs {
		allocs := testing.AllocsPerRun(1000, func() {
			var err error
			dst, err = tk.EncodeInto(input, dst[:0], &ws)
			if err != nil {
				panic(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("warmed official tokenizer EncodeInto(%q) allocated %.2f times/call, want 0", shortText(input), allocs)
		}
	}
}

func TestQwenTokenizerEncodeIntoZeroAllocAfterWarmup(t *testing.T) {
	tk := tinyQwenTokenizer()
	inputs := []string{
		"hello",
		"we're READY 2026!\r\n",
		"e\u0301 中文🙂 <|test|>!",
		strings.Repeat("token 12, ", 400),
	}
	dst := make([]int, 0, 8192)
	var ws TokenizerWorkspace
	for _, input := range inputs {
		out, err := tk.EncodeInto(input, dst[:0], &ws)
		if err != nil {
			t.Fatalf("warm %q: %v", shortText(input), err)
		}
		dst = out
	}
	for _, input := range inputs {
		out, err := tk.EncodeInto(input, dst[:0], &ws)
		if err != nil {
			t.Fatal(err)
		}
		dst = out
		allocs := testing.AllocsPerRun(1000, func() {
			var err error
			dst, err = tk.EncodeInto(input, dst[:0], &ws)
			if err != nil {
				panic(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("warmed EncodeInto(%q) allocated %.2f times/call, want 0", shortText(input), allocs)
		}
	}
}

func TestQwenTokenizerNeedsCallerTokenCapacity(t *testing.T) {
	tk := tinyQwenTokenizer()
	got, err := tk.EncodeInto("hello", make([]int, 0, 1), &TokenizerWorkspace{})
	if err != ErrTokenBufferSmall || got != nil {
		t.Fatalf("EncodeInto with short dst = (%v, %v), want (nil, ErrTokenBufferSmall)", got, err)
	}
}

func TestQwenNextPiece(t *testing.T) {
	cases := []struct {
		input string
		want  []string
	}{
		{"we're ready", []string{"we", "'re", " ready"}},
		{"  hello", []string{" ", " hello"}},
		{"12345", []string{"1", "2", "3", "4", "5"}},
		{"!!!\r\nX", []string{"!!!\r\n", "X"}},
		{"word\n\n", []string{"word", "\n\n"}},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			var got []string
			for i := 0; i < len(tc.input); {
				start, end := qwenNextPiece([]byte(tc.input), i, false)
				got = append(got, tc.input[start:end])
				i = end
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("pieces %q, want %q", got, tc.want)
			}
		})
	}
}

func qwenTokenizerDir(t *testing.T) string {
	t.Helper()
	return testmodels.Path(t, testmodels.Qwen3)
}

func tinyQwenTokenizer() *Tokenizer {
	t := &Tokenizer{merges: newMergeTable(2), trie: []qwenAddedNode{{tokenID: -1}}}
	for i := range t.byteID {
		t.byteID[i] = int32(i)
	}
	_ = t.addAddedToken("<|test|>", 300)
	// The globally lowest rank must merge first, after which the adjacent
	// (a,bc) candidate is exposed and resolves to the final ID 302.
	t.merges.insert(qwenPair('b', 'c'), qwenMerge{rank: 0, token: 301})
	t.merges.insert(qwenPair('a', 301), qwenMerge{rank: 1, token: 302})
	return t
}

func shortText(s string) string {
	if len(s) <= 80 {
		return s
	}
	return s[:77] + "..."
}
