// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"testing"

	"github.com/GetStream/gophonic/speech"
)

// Options.Languages limits detection: speech in an allowed language is
// transcribed as without the limit, and other speech is named as one of
// the allowed languages, never its own.
func TestLanguages(t *testing.T) {
	m := loadModel(t, FormatGPU)
	tr, err := NewTranscriber(m, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	ctx := context.Background()
	jfk, zh := clipPCM(t, "jfk"), clipPCM(t, "zh")
	var free, limited speech.Transcript
	if err := tr.Transcribe(ctx, jfk, speech.Options{}, &free); err != nil {
		t.Fatal(err)
	}
	if err := tr.Transcribe(ctx, jfk, speech.Options{Languages: []string{"pt", "en"}}, &limited); err != nil {
		t.Fatal(err)
	}
	if limited.Language != "English" || string(limited.Text) != string(free.Text) {
		t.Fatalf("limited to pt, en: %s %q; free: %s %q", limited.Language, limited.Text, free.Language, free.Text)
	}
	for _, c := range []struct {
		langs []string
		want  map[string]bool
	}{{[]string{"en", "pt"}, map[string]bool{"English": true, "Portuguese": true}}, {[]string{"zh", "en"}, map[string]bool{"Chinese": true}}} {
		if err := tr.Transcribe(ctx, zh, speech.Options{Languages: c.langs}, &limited); err != nil {
			t.Fatal(err)
		}
		t.Logf("Mandarin limited to %v: %s %q", c.langs, limited.Language, limited.Text)
		if !c.want[limited.Language] {
			t.Errorf("limited to %v: %s", c.langs, limited.Language)
		}
	}
	opts := speech.Options{Languages: []string{"pt", "en"}}
	if n := testing.AllocsPerRun(3, func() {
		if err := tr.Transcribe(ctx, jfk, opts, &limited); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("%v allocations per warm call", n)
	}
	if err := tr.Transcribe(ctx, jfk, speech.Options{Languages: []string{"en", "tlh"}}, &limited); err == nil {
		t.Error("an unknown language was accepted")
	}
}
