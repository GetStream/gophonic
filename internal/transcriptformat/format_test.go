// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package transcriptformat

import (
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
	"strings"
	"testing"
)

func TestAppendJSONStringControlBytes(t *testing.T) {
	input := []byte{'a', 0, 0x1b, '\n', '"', '\\'}
	input = append(input, []byte("\u2028")...)
	got := AppendJSONString(make([]byte, 0, 64), input)
	if strings.Contains(string(got), `\x`) {
		t.Fatalf("Go-only escape in JSON: %s", got)
	}
	var decoded string
	if err := vibejson.Unmarshal(got, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != string(input) {
		t.Fatalf("decoded %q, want %q", decoded, input)
	}
	allocs := testing.AllocsPerRun(100, func() { dst := make([]byte, 0, 64); AppendJSONString(dst, input) })
	if allocs != 0 {
		t.Fatalf("JSON appender: %g allocs/op", allocs)
	}
}

func TestAppendSubtitles(t *testing.T) {
	text := []byte(" hello")
	transcript := &speech.Transcript{Text: text, Segments: []speech.Segment{{Start: 0.5, End: 1.25, TextStart: 0, TextEnd: len(text)}}}
	srt := string(AppendSubtitles(make([]byte, 0, 128), transcript, false))
	if srt != "1\n00:00:00,500 --> 00:00:01,250\nhello\n\n" {
		t.Fatalf("SRT = %q", srt)
	}
	vtt := string(AppendSubtitles(make([]byte, 0, 128), transcript, true))
	if vtt != "WEBVTT\n\n00:00:00.500 --> 00:00:01.250\nhello\n\n" {
		t.Fatalf("VTT = %q", vtt)
	}
}
