// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/thesyncim/vibejson"
)

func TestEnglishTokenizerWhisperSpecialIDs(t *testing.T) {
	tok, err := NewTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	if tok.VocabSize() != VocabSize {
		t.Fatalf("vocab size %d, want %d", tok.VocabSize(), VocabSize)
	}
	checks := []struct {
		name string
		got  int
		want int
	}{
		{"EOT", tok.EOT(), 50256},
		{"SOT", tok.SOT(), 50257},
		{"transcribe", tok.Transcribe(), 50358},
		{"no speech", tok.NoSpeech(), 50361},
		{"no timestamps", tok.NoTimestamps(), 50362},
		{"timestamp begin", tok.TimestampBegin(), 50363},
	}
	for _, check := range checks {
		if check.got != check.want {
			t.Errorf("%s ID %d, want %d", check.name, check.got, check.want)
		}
	}
	if got, ok := tok.LanguageToken("en"); !ok || got != 50258 {
		t.Errorf("English language token = %d, %v; want 50258, true", got, ok)
	}
}

func TestTokenizerOfficialJFKTokenTranscript(t *testing.T) {
	tok, err := NewTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper", "jfk.oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Transcript string
		Tokens     []int
	}
	if err := vibejson.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	wantText := " " + oracle.Transcript
	encoded := make([]int, 0, len(wantText))
	encoded, err = tok.EncodeInto(encoded, wantText)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(encoded, oracle.Tokens) {
		t.Fatalf("JFK token IDs differ from the pinned oracle:\n got %v\nwant %v", encoded, oracle.Tokens)
	}
	decoded := make([]byte, 0, len(wantText))
	decoded, err = tok.DecodeInto(decoded, encoded)
	if err != nil {
		t.Fatal(err)
	}
	if string(decoded) != wantText {
		t.Fatalf("decoded JFK text %q, want %q", decoded, wantText)
	}
	if strings.TrimSpace(string(decoded)) != oracle.Transcript {
		t.Fatalf("trimmed JFK transcript %q, want %q", strings.TrimSpace(string(decoded)), oracle.Transcript)
	}
}

func TestTokenizerOfficialWhitespaceVectors(t *testing.T) {
	tok, err := NewTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	vectors := []struct {
		text string
		want []int
	}{
		{"  hello", []int{220, 23748}},
		{"a  b", []int{64, 220, 275}},
		{"a\n b", []int{64, 198, 275}},
	}
	for _, vector := range vectors {
		got := make([]int, 0, len(vector.text))
		got, err = tok.EncodeInto(got, vector.text)
		if err != nil {
			t.Fatalf("EncodeInto(%q): %v", vector.text, err)
		}
		if !reflect.DeepEqual(got, vector.want) {
			t.Errorf("EncodeInto(%q) = %v, want %v", vector.text, got, vector.want)
		}
	}
}

func TestMultilingualTokenizerSpecialIDs(t *testing.T) {
	tok, err := NewTokenizer(Multilingual)
	if err != nil {
		t.Fatal(err)
	}
	if tok.VocabSize() != 51865 {
		t.Fatalf("vocab size %d, want 51865", tok.VocabSize())
	}
	if tok.EOT() != 50257 || tok.SOT() != 50258 || tok.NoTimestamps() != 50363 || tok.TimestampBegin() != 50364 {
		t.Fatalf("unexpected multilingual token offsets: eot=%d sot=%d no_ts=%d timestamp=%d",
			tok.EOT(), tok.SOT(), tok.NoTimestamps(), tok.TimestampBegin())
	}
}

func TestTokenizerIntoSteadyStateAllocations(t *testing.T) {
	tok, err := NewTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	const text = " And so my fellow Americans ask not what your country can do for you."
	encoded := make([]int, 0, len(text))
	decoded := make([]byte, 0, len(text))
	var encodeErr, decodeErr error
	encodeAllocs := testing.AllocsPerRun(20, func() {
		encoded = encoded[:0]
		encoded, encodeErr = tok.EncodeInto(encoded, text)
	})
	if encodeErr != nil {
		t.Fatal(encodeErr)
	}
	decodeAllocs := testing.AllocsPerRun(20, func() {
		decoded = decoded[:0]
		decoded, decodeErr = tok.DecodeInto(decoded, encoded)
	})
	if decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if encodeAllocs != 0 || decodeAllocs != 0 {
		t.Fatalf("warm tokenizer allocations: encode %.2f, decode %.2f; want zero", encodeAllocs, decodeAllocs)
	}
}
