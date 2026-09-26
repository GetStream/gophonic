// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"fmt"
	"slices"
	"unicode"
	"unicode/utf8"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

// Transcripts limited to languages are limited to their writing systems:
// a call in English and Portuguese gets Latin letters, whatever a noise
// sounds like to the model. Languages written in a script outside this
// table limit nothing.
var scripts = map[speech.Language][]*unicode.RangeTable{
	speech.Chinese: {unicode.Han}, speech.Cantonese: {unicode.Han},
	speech.Japanese: {unicode.Han, unicode.Hiragana, unicode.Katakana},
	speech.Korean:   {unicode.Hangul, unicode.Han},
	speech.Russian:  {unicode.Cyrillic}, speech.Macedonian: {unicode.Cyrillic},
	speech.Arabic: {unicode.Arabic}, speech.Persian: {unicode.Arabic},
	speech.Hindi: {unicode.Devanagari}, speech.Thai: {unicode.Thai}, speech.Greek: {unicode.Greek},
	speech.English: {unicode.Latin}, speech.Portuguese: {unicode.Latin}, speech.Spanish: {unicode.Latin},
	speech.French: {unicode.Latin}, speech.German: {unicode.Latin}, speech.Italian: {unicode.Latin},
	speech.Dutch: {unicode.Latin}, speech.Indonesian: {unicode.Latin}, speech.Malay: {unicode.Latin},
	speech.Vietnamese: {unicode.Latin}, speech.Turkish: {unicode.Latin}, speech.Filipino: {unicode.Latin},
	speech.Swedish: {unicode.Latin}, speech.Danish: {unicode.Latin}, speech.Finnish: {unicode.Latin},
	speech.Polish: {unicode.Latin}, speech.Czech: {unicode.Latin},
	speech.Romanian: {unicode.Latin}, speech.Hungarian: {unicode.Latin},
}

// letterScripts are the scripts a token can belong to; a token with
// letters of none of the allowed ones is not written.
var letterScripts = []*unicode.RangeTable{unicode.Latin, unicode.Han, unicode.Hiragana, unicode.Katakana,
	unicode.Hangul, unicode.Cyrillic, unicode.Arabic, unicode.Devanagari, unicode.Thai, unicode.Greek,
	unicode.Hebrew, unicode.Bengali, unicode.Tamil, unicode.Telugu, unicode.Gujarati, unicode.Armenian,
	unicode.Georgian, unicode.Ethiopic, unicode.Khmer, unicode.Lao, unicode.Myanmar, unicode.Sinhala}

// limits are what limiting a transcript to a set of languages takes: the
// first tokens of their names, as the output gives its language (and of
// "None", for audio without speech), and which tokens their scripts may
// contain (nil: any). They are read-only, and shared by every lane.
type limits struct {
	names []int
	mask  []bool
}

// maxLimits bounds the sets a model keeps limits for; others are prepared
// at every call.
const maxLimits = 32

// limitsOf returns the limits of set, prepared at its first use.
func (m *Model) limitsOf(set speech.LanguageSet) (*limits, error) {
	m.limitsMu.RLock()
	l := m.limits[set]
	m.limitsMu.RUnlock()
	if l != nil {
		return l, nil
	}
	l = &limits{}
	var ws qwen3lm.TokenizerWorkspace
	for lang := range set.All() {
		if !m.languages.Has(lang) {
			return nil, fmt.Errorf("qwen3asr: language %v: %w", lang, speech.ErrUnsupported)
		}
		ids, err := m.tok.EncodeInto("language "+lang.Name(), make([]int, 0, 8), &ws)
		if err != nil || len(ids) < 2 || ids[0] != m.ids.language {
			return nil, fmt.Errorf("qwen3asr: language %v: %w", lang, speech.ErrUnsupported)
		}
		l.names = append(l.names, ids[1])
	}
	l.names = append(l.names, m.ids.none)
	if l.mask = m.scriptMask(set); l.mask != nil {
		l.mask[m.ids.eos[0]], l.mask[m.ids.eos[1]] = true, true
	}
	m.limitsMu.Lock()
	defer m.limitsMu.Unlock()
	if kept := m.limits[set]; kept != nil {
		return kept, nil
	}
	if len(m.limits) < maxLimits {
		if m.limits == nil {
			m.limits = map[speech.LanguageSet]*limits{}
		}
		m.limits[set] = l
	}
	return l, nil
}

// scriptMask returns, for each token, whether text in the writing systems
// of the languages of set may contain it, or nil if one of them has no
// known script.
func (m *Model) scriptMask(set speech.LanguageSet) []bool {
	var allow []*unicode.RangeTable
	for l := range set.All() {
		s, ok := scripts[l]
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
