// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"testing"
	"time"

	"github.com/GetStream/gophonic/speech"
)

// Transcribing speech while it is spoken, each call continuing the last
// one's transcript, ends with the transcript of the whole audio in one call;
// so does continuing a wrong transcript, or one in another language.
func TestPartialTranscriptsContinueToTheOfflineResult(t *testing.T) {
	for _, format := range []string{FormatGPU, FormatF16} {
		t.Run(format, func(t *testing.T) {
			tr, err := NewTranscriber(loadModel(t, format), 0)
			if err != nil {
				t.Fatal(err)
			}
			defer tr.Close()
			ctx := context.Background()
			for _, clip := range []string{"jfk", "zh"} {
				pcm := clipPCM(t, clip)
				var offline speech.Transcript
				if err := tr.Transcribe(ctx, pcm, speech.Options{}, &offline); err != nil {
					t.Fatal(err)
				}
				began := time.Now()
				if err := tr.Transcribe(ctx, pcm, speech.Options{}, &offline); err != nil {
					t.Fatal(err)
				}
				offlineTime := time.Since(began)
				// Half a second at a time, as speech arrives.
				var prev, cur speech.Transcript
				step := sampleRate / 2
				for n := step; n < len(pcm); n += step {
					if err := tr.Transcribe(ctx, pcm[:n], speech.Options{Partial: &prev}, &cur); err != nil {
						t.Fatal(err)
					}
					prev, cur = cur, prev
				}
				began = time.Now()
				if err := tr.Transcribe(ctx, pcm, speech.Options{Partial: &prev}, &cur); err != nil {
					t.Fatal(err)
				}
				final := time.Since(began)
				if string(cur.Text) != string(offline.Text) || cur.Language != offline.Language {
					t.Fatalf("%s: continued %q (%s), offline %q (%s)", clip, cur.Text, cur.Language, offline.Text, offline.Language)
				}
				t.Logf("%s: last pass %v, offline %v: %q", clip, final.Round(time.Millisecond), offlineTime.Round(time.Millisecond), cur.Text)
				// Warm continuations allocate nothing.
				partial := prev
				if n := testing.AllocsPerRun(3, func() {
					if err := tr.Transcribe(ctx, pcm, speech.Options{Partial: &partial}, &cur); err != nil {
						t.Fatal(err)
					}
				}); n != 0 {
					t.Fatalf("%s: a warm continuation allocated %.1f times", clip, n)
				}
				for _, wrong := range []speech.Transcript{
					{Text: []byte("Completely different words than anyone said."), Language: "English"},
					{Text: []byte("完全不同的话。"), Language: "Chinese"},
					{Text: offline.Text, Language: "French"},
				} {
					if err := tr.Transcribe(ctx, pcm, speech.Options{Partial: &wrong}, &cur); err != nil {
						t.Fatal(err)
					}
					if string(cur.Text) != string(offline.Text) || cur.Language != offline.Language {
						t.Fatalf("%s: continuing %q (%s) gave %q (%s), want %q", clip, wrong.Text, wrong.Language, cur.Text, cur.Language, offline.Text)
					}
				}
			}
		})
	}
}
