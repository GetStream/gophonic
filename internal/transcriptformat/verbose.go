// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package transcriptformat

import (
	"bytes"
	"strconv"

	"github.com/GetStream/gophonic/speech"
)

// AppendVerboseJSON appends OpenAI's verbose_json transcription object for t,
// with words when includeWords is set. samples is the 16 kHz duration.
func AppendVerboseJSON(dst []byte, t *speech.Transcript, includeWords bool, samples int) []byte {
	dst = append(dst, `{"task":"transcribe","language":"`...)
	name := t.Language.Name()
	for i := range len(name) {
		c := name[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		dst = append(dst, c)
	}
	dst = append(dst, `","duration":`...)
	dst = strconv.AppendFloat(dst, float64(samples)/16000, 'f', 3, 64)
	dst = append(dst, `,"text":`...)
	dst = AppendJSONString(dst, bytes.TrimSpace(t.Text))
	dst = append(dst, `,"segments":[`...)
	for i, segment := range t.Segments {
		if i != 0 {
			dst = append(dst, ',')
		}
		dst = append(dst, `{"id":`...)
		dst = strconv.AppendInt(dst, int64(i), 10)
		dst = append(dst, `,"start":`...)
		dst = strconv.AppendFloat(dst, segment.Start, 'f', 2, 64)
		dst = append(dst, `,"end":`...)
		dst = strconv.AppendFloat(dst, segment.End, 'f', 2, 64)
		dst = append(dst, `,"text":`...)
		dst = AppendJSONString(dst, t.Text[segment.TextStart:segment.TextEnd])
		dst = append(dst, '}')
	}
	dst = append(dst, ']')
	if includeWords {
		dst = append(dst, `,"words":[`...)
		for i, word := range t.Words {
			if i != 0 {
				dst = append(dst, ',')
			}
			dst = append(dst, `{"word":`...)
			dst = AppendJSONString(dst, t.Text[word.TextStart:word.TextEnd])
			dst = append(dst, `,"start":`...)
			dst = strconv.AppendFloat(dst, word.Start, 'f', 2, 64)
			dst = append(dst, `,"end":`...)
			dst = strconv.AppendFloat(dst, word.End, 'f', 2, 64)
			dst = append(dst, '}')
		}
		dst = append(dst, ']')
	}
	return append(dst, '}', '\n')
}
