// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"reflect"
	"strings"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

// Qwen3-ASR ships a slow Qwen2Tokenizer: vocab.json, merges.txt, and added
// tokens in tokenizer_config.json. The goldens are the official processor's
// prompt ids and the greedy continuations of the FP32 PyTorch model.
func TestQwen2TokenizerFilesMatchHF(t *testing.T) {
	tk, err := LoadTokenizer(testmodels.Path(t, testmodels.Qwen3ASR))
	if err != nil {
		t.Fatal(err)
	}
	prompt := "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
	want := []int{151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669, 151676, 151676, 151670, 151645, 198, 151644, 77091, 198}
	var ws TokenizerWorkspace
	got, err := tk.EncodeInto(prompt, make([]int, 0, 64), &ws)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("prompt ids %v, want %v", got, want)
	}
	for _, tc := range []struct {
		ids []int
		raw string
	}{
		{[]int{11528, 6364, 151704, 3036, 773, 11, 847, 12357, 8877, 11, 2548, 537, 1128, 697, 3146, 646, 653, 369, 498, 13, 20437, 1128, 498, 646, 653, 369, 697, 3146, 13, 151645},
			"language English<asr_text>And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."},
		{[]int{11528, 8453, 151704, 100636, 100347, 99886, 100740, 118083, 102072, 1773, 151645},
			"language Chinese<asr_text>甚至出现交易几乎停滞的情况。"},
	} {
		if got := string(tk.DecodeAppend(nil, tc.ids, true)); got != tc.raw {
			t.Errorf("decode %v = %q, want %q", tc.ids, got, tc.raw)
		}
		// The decoded text encodes back to the same ids.
		ids, err := tk.EncodeInto(tc.raw, make([]int, 0, 256), &ws)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ids, tc.ids[:len(tc.ids)-1]) {
			t.Errorf("encode %q = %v, want %v", tc.raw, ids, tc.ids[:len(tc.ids)-1])
		}
	}
	for content, id := range map[string]int{"<|im_end|>": 151645, "<|endoftext|>": 151643, "<asr_text>": 151704, "<|audio_pad|>": 151676} {
		if got, ok := tk.AddedID(content); !ok || got != id {
			t.Errorf("AddedID(%q) = %d, %t; want %d", content, got, ok, id)
		}
	}
	if _, ok := tk.AddedID("<|im_end|>x"); ok {
		t.Error("AddedID matched a token prefix")
	}
	if !tk.Special(151645) || tk.Special(151704) || tk.Special(11528) {
		t.Error("special flags differ from tokenizer_config.json")
	}
	if got := string(tk.DecodeAppend(nil, []int{151644, 872}, false)); got != "<|im_start|>user" {
		t.Errorf("decode with special tokens = %q", got)
	}
}

// The two file layouts describe one tokenizer: Qwen3-8B's tokenizer.json and
// Qwen3-ASR's slow-tokenizer files agree on every encoding.
func TestQwen2TokenizerFilesMatchTokenizerJSON(t *testing.T) {
	asr, err := LoadTokenizer(testmodels.Path(t, testmodels.Qwen3ASR))
	if err != nil {
		t.Fatal(err)
	}
	lm, err := LoadTokenizer(qwenTokenizerDir(t))
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("we're READY, I'M here! é café — 你好，世界🙂 ١٢٣ Ⅷ\r\n\t<|im_start|>x ", 20)
	var ws TokenizerWorkspace
	a, err := asr.EncodeInto(text, make([]int, 0, 4096), &ws)
	if err != nil {
		t.Fatal(err)
	}
	b, err := lm.EncodeInto(text, make([]int, 0, 4096), &ws)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("tokenizer.json and vocab.json/merges.txt encodings differ")
	}
	for id := range 151643 {
		if string(asr.Piece(id)) != string(lm.Piece(id)) {
			t.Fatalf("token %d decodes to %q and %q", id, asr.Piece(id), lm.Piece(id))
		}
	}
}

func TestDecodeReplacesInvalidUTF8LikePython(t *testing.T) {
	tk := &Tokenizer{}
	// Tokens 0..3 are raw byte fragments of "é" (C3 A9) and "€" (E2 82 AC).
	for _, piece := range []string{"\xc3", "\xa9", "\xe2\x82", "\xac"} {
		tk.pieces = append(tk.pieces, piece...)
		tk.pieceEnd = append(tk.pieceEnd, uint32(len(tk.pieces)))
	}
	tk.special = make([]bool, len(tk.pieceEnd))
	for _, tc := range []struct {
		ids  []int
		want string
	}{
		{[]int{0, 1}, "é"},
		{[]int{2, 3}, "€"},
		{[]int{0}, "�"},
		{[]int{2}, "�"}, // one maximal subpart, as Python decodes b"\xe2\x82"
		{[]int{1, 2, 0}, "���"},
		{[]int{3, 0, 1, 99}, "�é"}, // unknown ids decode to nothing
	} {
		prefix := []byte("ok:")
		if got := string(tk.DecodeAppend(prefix, tc.ids, true)); got != "ok:"+tc.want {
			t.Errorf("decode %v = %q, want %q", tc.ids, got, "ok:"+tc.want)
		}
	}
	for _, tc := range []struct{ in, want string }{
		{"\xed\xa0\x80", "���"}, // surrogates: ED A0 is not a valid prefix
		{"\xf0\x9f\x99", "�"},
		{"\xf4\x90\x80\x80", "����"},
		{"\xe0\x80", "��"},
		{"a\xffb", "a�b"},
	} {
		if got := string(appendReplacingInvalid(nil, []byte(tc.in))); got != tc.want {
			t.Errorf("replace %q = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestDecodeAppendDoesNotAllocate(t *testing.T) {
	tk, err := LoadTokenizer(testmodels.Path(t, testmodels.Qwen3ASR))
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{11528, 8453, 151704, 100636, 100347, 99886, 100740, 118083, 102072, 1773, 151645}
	buf := make([]byte, 0, 256)
	if n := testing.AllocsPerRun(100, func() { buf = tk.DecodeAppend(buf[:0], ids, true) }); n != 0 {
		t.Fatalf("DecodeAppend allocated %.1f times", n)
	}
}
