// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"math"
	"strings"
	"testing"
)

// The expected values are the reference package's detect_and_fix_repetitions
// and parse_asr_output applied to each raw string.
func TestParseMatchesReference(t *testing.T) {
	for _, tc := range []struct{ raw, fixed, language, text string }{
		{strings.Repeat("a", 25) + strings.Repeat("b", 20) + "c", "a" + strings.Repeat("b", 20) + "c", "", "a" + strings.Repeat("b", 20) + "c"},
		{strings.Repeat("ha", 30) + " end", "ha end", "", "ha end"},
		{"language English<asr_text>" + strings.Repeat("no ", 25) + "yes", "language English<asr_text>no yes", "English", "no yes"},
		{"language English<asr_text>" + strings.Repeat("好", 21) + "的", "language English<asr_text>好的", "English", "好的"},
		{"x" + strings.Repeat("abcdefghij", 21) + "tail" + strings.Repeat("zz", 20) + "!", "xabcdefghijtailz!", "", "xabcdefghijtailz!"},
		{"short", "short", "", "short"},
		{"language None<asr_text>", "language None<asr_text>", "", ""},
		{"language None<asr_text> leftover ", "language None<asr_text> leftover ", "", "leftover"},
		{"  language FRENCH\n<asr_text>  bonjour  ", "", "French", "bonjour"},
		{"  \n language german \n more <asr_text>hallo", "", "German", "hallo"},
		{"no tag at all  ", "", "", "no tag at all"},
		{"prefix\nlanguage spanish<asr_text>hola", "", "Spanish", "hola"},
		{strings.Repeat("abc", 19) + "abX", strings.Repeat("abc", 19) + "abX", "", strings.Repeat("abc", 19) + "abX"},
		{strings.Repeat("é", 21) + strings.Repeat("🙂", 40), "é🙂", "", "é🙂"},
		{"language Klingon<asr_text>nuqneH", "", "Klingon", "nuqneH"},
	} {
		if tc.fixed != "" {
			if got := string(fixPatternRepeats(fixCharRepeats([]rune(tc.raw)), nil)); got != tc.fixed {
				t.Errorf("fix(%q) = %q, want %q", tc.raw, got, tc.fixed)
			}
		}
		tr := &Transcriber{raw: []byte(tc.raw)}
		if language := tr.parse(""); language != tc.language || string(tr.text) != tc.text {
			t.Errorf("parse(%q) = %q, %q; want %q, %q", tc.raw, language, tr.text, tc.language, tc.text)
		}
	}
	tr := &Transcriber{raw: []byte("  forced text ")}
	if language := tr.parse("German"); language != "German" || string(tr.text) != "forced text" {
		t.Errorf("forced language: %q, %q", language, tr.text)
	}
}

// tokens follows the reference's _get_feat_extract_output_lengths.
func TestTokensMatchReferenceFormula(t *testing.T) {
	e := &encoder{chunkFrames: 100}
	floorDiv := func(a, b int) int { return int(math.Floor(float64(a) / float64(b))) }
	for frames := range 5000 {
		leave := frames % 100
		feat := floorDiv(leave-1, 2) + 1
		want := floorDiv(floorDiv(feat-1, 2)+1-1, 2) + 1 + frames/100*13
		if got := e.tokens(frames); got != want {
			t.Fatalf("tokens(%d) = %d, want %d", frames, got, want)
		}
	}
}

func TestSinusoids(t *testing.T) {
	const length, channels = 13, 1024
	table := sinusoids(length, channels)
	increment := math.Log(10000) / (channels/2 - 1)
	for p := range length {
		for i := range channels / 2 {
			angle := float64(p) * math.Exp(-increment*float64(i))
			if d := math.Abs(float64(table[p*channels+i]) - math.Sin(angle)); d > 2e-6 {
				t.Fatalf("sin(%d, %d) off by %g", p, i, d)
			}
			if d := math.Abs(float64(table[p*channels+channels/2+i]) - math.Cos(angle)); d > 2e-6 {
				t.Fatalf("cos(%d, %d) off by %g", p, i, d)
			}
		}
	}
}

// quietCut picks the quietest sample of the quietest 100 ms window within
// 5 s of the length limit, as the reference's split_audio_into_chunks does.
func TestQuietCut(t *testing.T) {
	pcm := make([]float32, maxChunkSamples+10*sampleRate)
	for i := range pcm {
		pcm[i] = float32(math.Sin(float64(i)*0.05)) * 0.5
	}
	gap := maxChunkSamples - 2*sampleRate // quiet 200 ms, 2 s before the limit
	for i := gap; i < gap+sampleRate/5; i++ {
		pcm[i] *= 0.001
	}
	pcm[gap+1000] = 0
	if cut := quietCut(pcm, 0); cut != gap+1000 {
		t.Fatalf("cut at %d, want %d", cut, gap+1000)
	}
	if cut := quietCut(pcm[:maxChunkSamples+10], 0); cut <= 0 || cut > maxChunkSamples+10 {
		t.Fatalf("cut %d outside the audio", cut)
	}
}
