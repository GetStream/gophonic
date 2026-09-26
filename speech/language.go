// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech

import (
	"fmt"
	"iter"
	"math/bits"
	"strings"
)

// Language is a spoken language known to at least one gophonic model. It
// is a small integer: comparable, free to pass, and the index of its bit in
// a LanguageSet. The zero value, Unknown, is no language in particular.
type Language uint8

// The languages, in the order of their English names.
const (
	Unknown Language = iota
	Arabic
	Cantonese
	Chinese
	Czech
	Danish
	Dutch
	English
	Filipino
	Finnish
	French
	German
	Greek
	Hindi
	Hungarian
	Indonesian
	Italian
	Japanese
	Korean
	Macedonian
	Malay
	Persian
	Polish
	Portuguese
	Romanian
	Russian
	Spanish
	Swedish
	Thai
	Turkish
	Vietnamese

	numLanguages
)

// A LanguageSet has a bit for every language.
const _ = uint64(1) << (numLanguages - 1)

var languageTable = [numLanguages]struct{ code, name string }{
	{"", ""},
	{"ar", "Arabic"}, {"yue", "Cantonese"}, {"zh", "Chinese"}, {"cs", "Czech"},
	{"da", "Danish"}, {"nl", "Dutch"}, {"en", "English"}, {"fil", "Filipino"},
	{"fi", "Finnish"}, {"fr", "French"}, {"de", "German"}, {"el", "Greek"},
	{"hi", "Hindi"}, {"hu", "Hungarian"}, {"id", "Indonesian"}, {"it", "Italian"},
	{"ja", "Japanese"}, {"ko", "Korean"}, {"mk", "Macedonian"}, {"ms", "Malay"},
	{"fa", "Persian"}, {"pl", "Polish"}, {"pt", "Portuguese"}, {"ro", "Romanian"},
	{"ru", "Russian"}, {"es", "Spanish"}, {"sv", "Swedish"}, {"th", "Thai"},
	{"tr", "Turkish"}, {"vi", "Vietnamese"},
}

// byName maps each language's code and name, in lower case, to it.
var byName = func() map[string]Language {
	m := make(map[string]Language, 2*numLanguages+1)
	for l := Language(1); l < numLanguages; l++ {
		m[languageTable[l].code] = l
		m[strings.ToLower(languageTable[l].name)] = l
	}
	m["tl"] = Filipino // ISO 639-1 for Tagalog, which Filipino is based on
	return m
}()

// ParseLanguage reads a language given by its ISO 639 code ("pt") or its
// English name ("Portuguese"), in any case. It does not allocate.
func ParseLanguage[S ~string | ~[]byte](s S) (Language, bool) {
	var lower [16]byte
	if len(s) > len(lower) {
		return Unknown, false
	}
	for i := range len(s) {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		lower[i] = c
	}
	l, ok := byName[string(lower[:len(s)])]
	return l, ok
}

// Code is the language's ISO 639 code, such as "en" or "yue"; Unknown has
// none.
func (l Language) Code() string {
	if l >= numLanguages {
		return ""
	}
	return languageTable[l].code
}

// Name is the language's English name, such as "English"; Unknown has
// none.
func (l Language) Name() string {
	if l >= numLanguages {
		return ""
	}
	return languageTable[l].name
}

// String is the language's name, or "unknown".
func (l Language) String() string {
	if name := l.Name(); name != "" {
		return name
	}
	return "unknown"
}

// LanguageSet is a set of languages, one bit each. It is comparable in one
// instruction, so a lane tells at once whether a call's languages are the
// ones it prepared for. The zero value is empty.
type LanguageSet uint64

// Languages is the set of l.
func Languages(l ...Language) LanguageSet {
	var s LanguageSet
	for _, x := range l {
		s = s.With(x)
	}
	return s
}

// ParseLanguages reads a comma-separated list of languages, each a code or
// a name: "en,pt" or "English, Portuguese".
func ParseLanguages(list string) (LanguageSet, error) {
	var s LanguageSet
	for field := range strings.SplitSeq(list, ",") {
		if field = strings.TrimSpace(field); field == "" {
			continue
		}
		l, ok := ParseLanguage(field)
		if !ok {
			return 0, fmt.Errorf("speech: unknown language %q: %w", field, ErrUnsupported)
		}
		s = s.With(l)
	}
	return s, nil
}

// Has reports whether l is in s.
func (s LanguageSet) Has(l Language) bool { return l != Unknown && l < numLanguages && s&(1<<l) != 0 }

// With is s and l.
func (s LanguageSet) With(l Language) LanguageSet {
	if l == Unknown || l >= numLanguages {
		return s
	}
	return s | 1<<l
}

// Len is the number of languages in s.
func (s LanguageSet) Len() int { return bits.OnesCount64(uint64(s)) }

// All yields the languages of s in order.
func (s LanguageSet) All() iter.Seq[Language] {
	return func(yield func(Language) bool) {
		for rest := uint64(s); rest != 0; rest &= rest - 1 {
			if !yield(Language(bits.TrailingZeros64(rest))) {
				return
			}
		}
	}
}

// String lists the codes of s: "en,pt".
func (s LanguageSet) String() string {
	var b strings.Builder
	for l := range s.All() {
		if b.Len() > 0 {
			b.WriteByte(',')
		}
		b.WriteString(l.Code())
	}
	return b.String()
}
