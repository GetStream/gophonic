// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"slices"
	"unicode"
	"unicode/utf8"
)

// Transcripts limited to languages are limited to their writing systems:
// a call in English and Portuguese gets Latin letters, whatever a noise
// sounds like to the model. Languages written in a script outside this
// table limit nothing.
var scripts = map[string][]*unicode.RangeTable{
	"Chinese": {unicode.Han}, "Cantonese": {unicode.Han},
	"Japanese": {unicode.Han, unicode.Hiragana, unicode.Katakana},
	"Korean":   {unicode.Hangul, unicode.Han},
	"Russian":  {unicode.Cyrillic}, "Ukrainian": {unicode.Cyrillic}, "Bulgarian": {unicode.Cyrillic},
	"Macedonian": {unicode.Cyrillic}, "Serbian": {unicode.Cyrillic, unicode.Latin},
	"Arabic": {unicode.Arabic}, "Persian": {unicode.Arabic}, "Urdu": {unicode.Arabic},
	"Hindi": {unicode.Devanagari}, "Marathi": {unicode.Devanagari}, "Nepali": {unicode.Devanagari},
	"Thai": {unicode.Thai}, "Greek": {unicode.Greek}, "Hebrew": {unicode.Hebrew},
	"English": {unicode.Latin}, "Portuguese": {unicode.Latin}, "Spanish": {unicode.Latin},
	"French": {unicode.Latin}, "German": {unicode.Latin}, "Italian": {unicode.Latin},
	"Dutch": {unicode.Latin}, "Indonesian": {unicode.Latin}, "Malay": {unicode.Latin},
	"Vietnamese": {unicode.Latin}, "Turkish": {unicode.Latin}, "Filipino": {unicode.Latin},
	"Swedish": {unicode.Latin}, "Danish": {unicode.Latin}, "Finnish": {unicode.Latin},
	"Norwegian": {unicode.Latin}, "Polish": {unicode.Latin}, "Czech": {unicode.Latin},
	"Romanian": {unicode.Latin}, "Hungarian": {unicode.Latin},
}

// letterScripts are the scripts a token can belong to; a token with
// letters of none of the allowed ones is not written.
var letterScripts = []*unicode.RangeTable{unicode.Latin, unicode.Han, unicode.Hiragana, unicode.Katakana,
	unicode.Hangul, unicode.Cyrillic, unicode.Arabic, unicode.Devanagari, unicode.Thai, unicode.Greek,
	unicode.Hebrew, unicode.Bengali, unicode.Tamil, unicode.Telugu, unicode.Gujarati, unicode.Armenian,
	unicode.Georgian, unicode.Ethiopic, unicode.Khmer, unicode.Lao, unicode.Myanmar, unicode.Sinhala}

// scriptMask returns, for each token, whether text in the writing systems
// of languages may contain it, or nil if one of them has no known script.
func (m *Model) scriptMask(languages []string) []bool {
	var allow []*unicode.RangeTable
	for _, name := range languages {
		s, ok := scripts[name]
		if !ok {
			return nil
		}
		for _, t := range s {
			if !slices.Contains(allow, t) {
				allow = append(allow, t)
			}
		}
	}
	mask := make([]bool, m.lm.Config().Vocab)
	for id := range mask {
		mask[id] = writes(m.tok.Piece(id), allow)
	}
	return mask
}

// writes reports whether a token's bytes can be part of text written in
// the allowed scripts. Bytes that start a multibyte character the token
// splits count as that character's block.
func writes(piece []byte, allow []*unicode.RangeTable) bool {
	for len(piece) > 0 {
		r, n := utf8.DecodeRune(piece)
		if r == utf8.RuneError && n <= 1 {
			if !leadAllowed(piece[0], allow) {
				return false
			}
			piece = piece[1:]
			continue
		}
		piece = piece[n:]
		if !unicode.IsLetter(r) {
			continue
		}
		for _, s := range letterScripts {
			if unicode.Is(s, r) && !slices.Contains(allow, s) {
				return false
			}
		}
	}
	return true
}

// leadAllowed reports whether a byte that starts a multibyte UTF-8
// character a token splits may start one in the allowed scripts: E3–E9
// start kana and Han, EA–ED Hangul, D0–D4 Cyrillic, D8–DB Arabic, and E0
// the Indic scripts and Thai; other bytes are punctuation, symbols, the
// Latin, Greek, and Hebrew blocks, or continuations.
func leadAllowed(b byte, allow []*unicode.RangeTable) bool {
	has := func(ts ...*unicode.RangeTable) bool {
		for _, t := range ts {
			if slices.Contains(allow, t) {
				return true
			}
		}
		return false
	}
	switch {
	case b >= 0xE3 && b <= 0xE9:
		return has(unicode.Han, unicode.Hiragana, unicode.Katakana)
	case b >= 0xEA && b <= 0xED:
		return has(unicode.Hangul)
	case b >= 0xD0 && b <= 0xD4:
		return has(unicode.Cyrillic)
	case b >= 0xD8 && b <= 0xDB:
		return has(unicode.Arabic)
	case b == 0xE0:
		return has(unicode.Devanagari, unicode.Thai, unicode.Bengali, unicode.Tamil, unicode.Telugu, unicode.Gujarati, unicode.Sinhala, unicode.Lao, unicode.Khmer, unicode.Myanmar)
	}
	return true
}
