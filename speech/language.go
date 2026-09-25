// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech

import "strings"

// Language is a spoken language known to at least one gophonic model.
type Language struct {
	Code string // ISO 639 code, such as "en" or "yue"
	Name string // English name, as Transcript.Language reports it
}

// Languages lists the languages gophonic's models can be asked for. Each
// model supports a subset and rejects others with ErrUnsupported.
var Languages = []Language{
	{"ar", "Arabic"}, {"yue", "Cantonese"}, {"zh", "Chinese"}, {"cs", "Czech"},
	{"da", "Danish"}, {"nl", "Dutch"}, {"en", "English"}, {"fil", "Filipino"},
	{"fi", "Finnish"}, {"fr", "French"}, {"de", "German"}, {"el", "Greek"},
	{"hi", "Hindi"}, {"hu", "Hungarian"}, {"id", "Indonesian"}, {"it", "Italian"},
	{"ja", "Japanese"}, {"ko", "Korean"}, {"mk", "Macedonian"}, {"ms", "Malay"},
	{"fa", "Persian"}, {"pl", "Polish"}, {"pt", "Portuguese"}, {"ro", "Romanian"},
	{"ru", "Russian"}, {"es", "Spanish"}, {"sv", "Swedish"}, {"th", "Thai"},
	{"tr", "Turkish"}, {"vi", "Vietnamese"},
}

var languageIndex = func() map[string]string {
	m := make(map[string]string, 3*len(Languages))
	for _, l := range Languages {
		m[l.Code] = l.Name
		m[l.Name] = l.Name
		m[strings.ToLower(l.Name)] = l.Name
	}
	m["tl"] = "Filipino" // ISO 639-1 for Tagalog, which Filipino is based on
	return m
}()

// LanguageName returns the English name of a language given by code or name
// ("de", "German", or "german"). It does not allocate.
func LanguageName(language string) (string, bool) {
	name, ok := languageIndex[language]
	return name, ok
}

// LanguageNameBytes is LanguageName for a byte slice; it does not allocate.
func LanguageNameBytes(language []byte) (string, bool) {
	name, ok := languageIndex[string(language)]
	return name, ok
}
