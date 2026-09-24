// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package transcriptformat

import (
	"github.com/GetStream/gophonic/whisper"
	"strconv"
)

func AppendVerboseJSON(dst, text, rawText []byte, segments []whisper.Segment, words []whisper.Word, includeWords bool, samples int) []byte {
	dst = append(dst, `{"task":"transcribe","language":"english","duration":`...)
	dst = strconv.AppendFloat(dst, float64(samples)/16000, 'f', 3, 64)
	dst = append(dst, `,"text":`...)
	dst = AppendJSONString(dst, text)
	dst = append(dst, `,"segments":[`...)
	for i, segment := range segments {
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
		dst = AppendJSONString(dst, rawText[segment.TextStart:segment.TextEnd])
		dst = append(dst, '}')
	}
	dst = append(dst, ']')
	if includeWords {
		dst = append(dst, `,"words":[`...)
		for i, word := range words {
			if i != 0 {
				dst = append(dst, ',')
			}
			dst = append(dst, `{"word":`...)
			dst = AppendJSONString(dst, rawText[word.TextStart:word.TextEnd])
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
