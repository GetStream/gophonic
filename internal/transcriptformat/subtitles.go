// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package transcriptformat

import (
	"bytes"
	"math"
	"strconv"

	"github.com/GetStream/gophonic/speech"
)

func AppendSubtitles(dst []byte, t *speech.Transcript, vtt bool) []byte {
	if vtt {
		dst = append(dst, "WEBVTT\n\n"...)
	}
	for i, segment := range t.Segments {
		if !vtt {
			dst = strconv.AppendInt(dst, int64(i+1), 10)
			dst = append(dst, '\n')
		}
		dst = appendSubtitleTime(dst, segment.Start, vtt)
		dst = append(dst, " --> "...)
		dst = appendSubtitleTime(dst, segment.End, vtt)
		dst = append(dst, '\n')
		dst = append(dst, bytes.TrimSpace(t.Text[segment.TextStart:segment.TextEnd])...)
		dst = append(dst, '\n', '\n')
	}
	return dst
}

func appendSubtitleTime(dst []byte, seconds float64, vtt bool) []byte {
	millis := int64(math.Round(seconds * 1000))
	if millis < 0 {
		millis = 0
	}
	hours := millis / 3600000
	minutes := (millis / 60000) % 60
	secs := (millis / 1000) % 60
	ms := millis % 1000
	dst = appendPadded(dst, hours, 2)
	dst = append(dst, ':')
	dst = appendPadded(dst, minutes, 2)
	dst = append(dst, ':')
	dst = appendPadded(dst, secs, 2)
	if vtt {
		dst = append(dst, '.')
	} else {
		dst = append(dst, ',')
	}
	return appendPadded(dst, ms, 3)
}

func appendPadded(dst []byte, value int64, width int) []byte {
	var scratch [20]byte
	digits := strconv.AppendInt(scratch[:0], value, 10)
	for i := len(digits); i < width; i++ {
		dst = append(dst, '0')
	}
	return append(dst, digits...)
}
