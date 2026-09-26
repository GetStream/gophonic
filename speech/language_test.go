// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package speech

import (
	"errors"
	"slices"
	"testing"
)

func TestParseLanguage(t *testing.T) {
	for _, c := range []struct {
		in   string
		want Language
		ok   bool
	}{
		{"pt", Portuguese, true}, {"Portuguese", Portuguese, true}, {"PORTUGUESE", Portuguese, true},
		{"yue", Cantonese, true}, {"tl", Filipino, true}, {"fil", Filipino, true},
		{"", Unknown, false}, {"tlh", Unknown, false}, {"Klingon", Unknown, false},
		{"a language with a very long name", Unknown, false},
	} {
		if got, ok := ParseLanguage(c.in); got != c.want || ok != c.ok {
			t.Errorf("ParseLanguage(%q) = %v, %v; want %v, %v", c.in, got, ok, c.want, c.ok)
		}
		if got, ok := ParseLanguage([]byte(c.in)); got != c.want || ok != c.ok {
			t.Errorf("ParseLanguage([]byte(%q)) = %v, %v", c.in, got, ok)
		}
	}
	if English.Code() != "en" || English.Name() != "English" || English.String() != "English" || Unknown.String() != "unknown" {
		t.Fatalf("English is %q, %q, %q; Unknown %q", English.Code(), English.Name(), English.String(), Unknown.String())
	}
	b := []byte("German")
	if n := testing.AllocsPerRun(100, func() {
		ParseLanguage("de")
		ParseLanguage(b)
	}); n != 0 {
		t.Fatalf("ParseLanguage allocates %v times", n)
	}
}

func TestLanguageSet(t *testing.T) {
	s := Languages(Portuguese, English, Unknown)
	if !s.Has(English) || !s.Has(Portuguese) || s.Has(German) || s.Has(Unknown) || s.Len() != 2 {
		t.Fatalf("set %v", s)
	}
	if got := slices.Collect(s.All()); !slices.Equal(got, []Language{English, Portuguese}) || s.String() != "en,pt" {
		t.Fatalf("All %v, String %q", got, s.String())
	}
	if s.With(German).Len() != 3 || s != Languages(English, Portuguese) {
		t.Fatal("With changed the set it was called on, or equal sets differ")
	}
	if all := Languages(Arabic, Vietnamese); !all.Has(Arabic) || !all.Has(Vietnamese) {
		t.Fatal("the first and last languages have no bit")
	}
	if p, err := ParseLanguages("en, Portuguese,,"); err != nil || p != s {
		t.Fatalf("ParseLanguages: %v, %v", p, err)
	}
	if _, err := ParseLanguages("en,tlh"); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("ParseLanguages of an unknown language: %v", err)
	}
}
