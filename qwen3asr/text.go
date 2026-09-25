// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"bytes"
	"unicode"
	"unicode/utf8"

	"github.com/GetStream/gophonic/speech"
)

// The model answers "language English<asr_text>text", "language None" for
// audio without speech, or the text alone when the prompt forced the
// language. parse follows the reference's parse_asr_output on t.raw: trim,
// remove runaway repetitions, split off the language, and trim the text.

// repeatThreshold is the reference's detect_and_fix_repetitions threshold:
// a character or a pattern of up to maxPattern characters repeated more
// often collapses to one occurrence.
const (
	repeatThreshold = 20
	maxPattern      = 20
)

// parse writes the transcript text of t.raw to t.text and returns the
// language: forced, when the caller forced one, otherwise the one the model
// named, or empty.
func (t *Transcriber) parse(forced string) string {
	s := bytes.TrimFunc(t.raw, isSpace)
	if len(s) == 0 {
		return forced
	}
	t.runes = t.runes[:0]
	for len(s) > 0 {
		r, n := utf8.DecodeRune(s)
		t.runes = append(t.runes, r)
		s = s[n:]
	}
	t.fixed = fixPatternRepeats(fixCharRepeats(t.runes), t.fixed[:0])
	t.text = t.text[:0]
	for _, r := range t.fixed {
		t.text = utf8.AppendRune(t.text, r)
	}
	if forced != "" {
		return forced
	}
	meta, text, tagged := bytes.Cut(t.text, []byte("<asr_text>"))
	if !tagged {
		t.text = bytes.TrimFunc(t.text, isSpace)
		return ""
	}
	language := ""
	if containsFold(meta, "language none") {
		meta = nil
	}
	for line := range bytes.Lines(meta) {
		line = bytes.TrimFunc(line, isSpace)
		if len(line) == 0 {
			continue
		}
		if len(line) >= len("language ") && hasPrefixFold(line, "language ") {
			if value := bytes.TrimFunc(line[len("language "):], isSpace); len(value) > 0 {
				language = t.languageName(value)
			}
			break
		}
	}
	text = bytes.TrimFunc(text, isSpace)
	t.text = t.text[:copy(t.text, text)]
	return language
}

// languageName maps the model's language name to speech's English name
// without allocating; a name outside speech.Languages is kept as written,
// capitalized like the reference's normalize_language_name.
func (t *Transcriber) languageName(value []byte) string {
	if name, ok := speech.LanguageNameBytes(value); ok {
		return name
	}
	var lower [32]byte
	if len(value) <= len(lower) {
		for i, b := range value {
			if b >= 'A' && b <= 'Z' {
				b += 'a' - 'A'
			}
			lower[i] = b
		}
		if name, ok := speech.LanguageNameBytes(lower[:len(value)]); ok {
			return name
		}
	}
	r, n := utf8.DecodeRune(value)
	return string(unicode.ToUpper(r)) + string(bytes.ToLower(value[n:]))
}

// isSpace matches the characters Python's str.strip removes.
func isSpace(r rune) bool {
	return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f
}

func hasPrefixFold(s []byte, prefix string) bool {
	for i := range len(prefix) {
		if s[i]|0x20 != prefix[i]|0x20 || (prefix[i] == ' ') != (s[i] == ' ') {
			return false
		}
	}
	return true
}

func containsFold(s []byte, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if hasPrefixFold(s[i:], sub) {
			return true
		}
	}
	return false
}

// fixCharRepeats collapses each run of more than repeatThreshold equal
// characters to one, in place.
func fixCharRepeats(s []rune) []rune {
	out := s[:0]
	for i := 0; i < len(s); {
		n := 1
		for i+n < len(s) && s[i+n] == s[i] {
			n++
		}
		if n > repeatThreshold {
			out = append(out, s[i])
		} else {
			out = append(out, s[i:i+n]...)
		}
		i += n
	}
	return out
}

// fixPatternRepeats appends s to out with the first pattern of up to
// maxPattern characters repeated at least repeatThreshold times collapsed
// to one occurrence, then treats the rest of s the same way.
func fixPatternRepeats(s, out []rune) []rune {
	for {
		n := len(s)
		if n < 2*repeatThreshold {
			return append(out, s...)
		}
		found := false
		i := 0
		for ; i <= n-2*repeatThreshold; i++ {
			if k := patternAt(s[i:]); k > 0 {
				end := i + k*repeatThreshold
				for end+k <= n && equal(s[end:end+k], s[i:i+k]) {
					end += k
				}
				out = append(out, s[i:i+k]...)
				s, found = s[end:], true
				break
			}
			out = append(out, s[i])
		}
		if !found {
			return append(out, s[i:]...)
		}
	}
}

// patternAt returns the length of the shortest pattern that s starts with
// repeatThreshold times, or zero.
func patternAt(s []rune) int {
	for k := 1; k <= maxPattern && k*repeatThreshold <= len(s); k++ {
		if repeats(s, k) {
			return k
		}
	}
	return 0
}

// repeats reports whether s starts with repeatThreshold copies of its first
// k characters.
func repeats(s []rune, k int) bool {
	for rep := 1; rep < repeatThreshold; rep++ {
		if !equal(s[rep*k:rep*k+k], s[:k]) {
			return false
		}
	}
	return true
}

func equal(a, b []rune) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
