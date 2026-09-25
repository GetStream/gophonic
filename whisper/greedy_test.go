// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"reflect"
	"testing"

	"github.com/thesyncim/vibejson"
)

func TestGreedyEnglishPromptAndOfficialSuppression(t *testing.T) {
	tok, err := NewTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := NewGreedyPolicy(tok, GreedyOptions{WithoutTimestamps: true})
	if err != nil {
		t.Fatal(err)
	}
	prompt := make([]int, 0, TextContext)
	sampleBegin, err := policy.PromptInto(prompt, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	prompt = prompt[:sampleBegin]
	if !reflect.DeepEqual(prompt, []int{tok.SOT(), tok.NoTimestamps()}) {
		t.Fatalf("tiny.en initial prompt %v, want [%d %d]", prompt, tok.SOT(), tok.NoTimestamps())
	}

	nonSpeech, err := nonSpeechTokenIDs(tok)
	if err != nil {
		t.Fatal(err)
	}
	if len(nonSpeech) != 84 {
		t.Fatalf("non-speech suppression count %d, want 84", len(nonSpeech))
	}
	var suppressIDs []int
	for id, suppress := range policy.suppressed {
		if suppress {
			suppressIDs = append(suppressIDs, id)
		}
	}
	if len(suppressIDs) != 90 {
		t.Fatalf("combined non-speech and control suppression count %d, want 90", len(suppressIDs))
	}
	encoded, err := vibejson.Marshal(&suppressIDs)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(encoded)
	if got := hex.EncodeToString(sum[:]); got != "d816e35722bfd9dfd451be5840209d6f78f7bfce88ac395a42e96729e1a0ca3d" {
		t.Fatalf("official suppression IDs SHA-256 %s, want pinned checksum; ids=%v", got, suppressIDs)
	}

	logits := make([]float32, tok.VocabSize())
	for i := range logits {
		logits[i] = -10
	}
	logits[843] = 10
	logits[tok.EOT()] = 100
	logits[220] = 99
	logits[tok.Transcribe()] = 98
	filtered := make([]float32, tok.VocabSize())
	next, err := policy.SelectNextInto(filtered, logits, prompt)
	if err != nil {
		t.Fatal(err)
	}
	if next != 843 {
		t.Fatalf("first selected token %d, want 843", next)
	}
	for _, id := range []int{tok.EOT(), 220, tok.Transcribe()} {
		if !math.IsInf(float64(filtered[id]), -1) {
			t.Errorf("first-step token %d remained selectable: %g", id, filtered[id])
		}
	}
	if logits[tok.EOT()] != 100 {
		t.Fatal("SelectNextInto changed the source logits")
	}
}

func TestGreedyTimestampRuleAndNoTimestampMode(t *testing.T) {
	tok, err := NewTokenizer(Multilingual)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := NewGreedyPolicy(tok, GreedyOptions{DisableNonSpeechSuppression: true})
	if err != nil {
		t.Fatal(err)
	}
	history := make([]int, 0, TextContext)
	n, err := policy.PromptInto(history, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	history = history[:n]
	wantPrefix := []int{tok.SOT()}
	language, ok := tok.LanguageToken("en")
	if !ok {
		t.Fatal("English language token missing")
	}
	wantPrefix = append(wantPrefix, language, tok.Transcribe())
	if !reflect.DeepEqual(history, wantPrefix) {
		t.Fatalf("multilingual prompt %v, want %v", history, wantPrefix)
	}

	logits := make([]float32, tok.VocabSize())
	for i := range logits {
		logits[i] = -10
	}
	begin := tok.TimestampBegin()
	logits[0] = 100 // Text is masked on the first timestamped step.
	logits[begin+10] = 5
	logits[begin+51] = 50 // Beyond the default one second initial limit.
	filtered := make([]float32, tok.VocabSize())
	next, err := policy.SelectNextInto(filtered, logits, history)
	if err != nil {
		t.Fatal(err)
	}
	if next != begin+10 {
		t.Fatalf("first timestamp token %d, want %d", next, begin+10)
	}
	if !math.IsInf(float64(filtered[begin+51]), -1) {
		t.Fatal("initial timestamp limit did not mask timestamps after one second")
	}

	noTimestamp, err := NewGreedyPolicy(tok, GreedyOptions{WithoutTimestamps: true, DisableNonSpeechSuppression: true})
	if err != nil {
		t.Fatal(err)
	}
	plain := make([]int, 0, TextContext)
	plainN, err := noTimestamp.PromptInto(plain, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	plain = plain[:plainN]
	if !reflect.DeepEqual(plain, append(append([]int(nil), wantPrefix...), tok.NoTimestamps())) {
		t.Fatalf("without-timestamps prompt %v", plain)
	}
	for i := range logits {
		logits[i] = -10
	}
	logits[begin+10] = 5
	filtered = filtered[:0]
	filtered = make([]float32, tok.VocabSize())
	next, err = noTimestamp.SelectNextInto(filtered, logits, plain)
	if err != nil {
		t.Fatal(err)
	}
	if next != begin+10 || math.IsInf(float64(filtered[begin+10]), -1) {
		t.Fatalf("without_timestamps unexpectedly masked timestamp token: next=%d, logit=%g", next, filtered[begin+10])
	}
}

func TestGreedyEnglishTimestampMode(t *testing.T) {
	tok, err := NewTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := NewGreedyPolicy(tok, GreedyOptions{DisableNonSpeechSuppression: true})
	if err != nil {
		t.Fatal(err)
	}
	history := make([]int, 0, TextContext)
	n, err := policy.PromptInto(history, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	history = history[:n]
	if !reflect.DeepEqual(history, []int{tok.SOT()}) {
		t.Fatalf("timestamp-enabled tiny.en prompt %v, want [%d]", history, tok.SOT())
	}

	logits := make([]float32, tok.VocabSize())
	for i := range logits {
		logits[i] = -10
	}
	begin := tok.TimestampBegin()
	logits[0] = 100
	logits[begin+10] = 5
	logits[begin+51] = 50
	filtered := make([]float32, tok.VocabSize())
	next, err := policy.SelectNextInto(filtered, logits, history)
	if err != nil {
		t.Fatal(err)
	}
	if next != begin+10 {
		t.Fatalf("first timestamp token %d, want %d", next, begin+10)
	}
	if !math.IsInf(float64(filtered[begin+51]), -1) {
		t.Fatal("first timestamp limit did not mask timestamps after one second")
	}

	history = append(history, next)
	for i := range logits {
		logits[i] = -10
	}
	logits[0] = 5
	logits[begin+11] = 10
	next, err = policy.SelectNextInto(filtered, logits, history)
	if err != nil {
		t.Fatal(err)
	}
	if next != 0 {
		t.Fatalf("token following initial timestamp %d, want text token 0", next)
	}
	if !math.IsInf(float64(filtered[begin+11]), -1) {
		t.Fatal("timestamp following an unmatched initial timestamp remained selectable")
	}
}

func TestGreedySteadyStateAllocations(t *testing.T) {
	tok, err := NewTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := NewGreedyPolicy(tok, GreedyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	history := make([]int, 0, TextContext)
	n, err := policy.PromptInto(history, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	history = history[:n]
	logits := make([]float32, tok.VocabSize())
	filtered := make([]float32, tok.VocabSize())
	var next int
	var selectErr error
	allocs := testing.AllocsPerRun(20, func() {
		next, selectErr = policy.SelectNextInto(filtered, logits, history)
	})
	if selectErr != nil {
		t.Fatal(selectErr)
	}
	if next < 0 {
		t.Fatalf("invalid selected token %d", next)
	}
	if allocs != 0 {
		t.Fatalf("warm SelectNextInto allocated %.2f objects, want zero", allocs)
	}
}
