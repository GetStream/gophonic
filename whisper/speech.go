// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"context"
	"fmt"

	"github.com/GetStream/gophonic/speech"
)

var _ speech.Transcriber = (*Transcriber)(nil)

// Transcribe implements speech.Transcriber with the whole-file path of
// TranscribeInto, TranscribeSegmentsInto, or TranscribeWordsInto. The English
// models accept an empty or English Language and no Context. dst's slices
// grow once to fit the audio; warm calls on audio of the same length or
// shorter do not allocate.
func (t *Transcriber) Transcribe(ctx context.Context, pcm []float32, opts speech.Options, dst *speech.Transcript) error {
	if t == nil || t.closed {
		return ErrTranscriberClosed
	}
	if l := opts.Language; l != speech.Unknown && l != speech.English {
		return fmt.Errorf("whisper: English model cannot transcribe %v: %w", l, speech.ErrUnsupported)
	}
	if opts.Context != "" {
		return fmt.Errorf("whisper: context prompts: %w", speech.ErrUnsupported)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	dst.Reset()
	dst.Text = grow(dst.Text, max(4096, len(pcm)/16))
	mode := uint8(0)
	if opts.Segments || opts.Words {
		mode = 1
		t.segments = grow(t.segments, max(64, len(pcm)/2000))
	}
	if opts.Words {
		mode = 2
		t.words = grow(t.words, max(128, len(pcm)/200))
	}
	text, segments, words, err := t.transcribeFullInto(pcm, dst.Text, t.segments[:0], t.words[:0], mode)
	if err != nil {
		return err
	}
	dst.Text = text
	dst.Language = speech.English
	for _, s := range segments {
		dst.Segments = append(dst.Segments, speech.Segment{Start: s.Start, End: s.End,
			TextStart: s.TextStart, TextEnd: s.TextEnd, WordStart: s.WordStart, WordEnd: s.WordEnd})
	}
	for _, w := range words {
		dst.Words = append(dst.Words, speech.Word(w))
	}
	return nil
}

// grow returns s emptied, with capacity for at least n elements.
func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, 0, n)
	}
	return s[:0]
}
