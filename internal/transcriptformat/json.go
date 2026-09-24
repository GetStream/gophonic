// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package transcriptformat

import "unicode/utf8"

// appendJSONString writes RFC 8259 string escaping into caller-owned storage.
// Unlike strconv.AppendQuote, it never emits Go-only \\x escapes.
func AppendJSONString(dst, src []byte) []byte {
	const hex = "0123456789abcdef"
	dst = append(dst, '"')
	for len(src) > 0 {
		b := src[0]
		switch b {
		case '"', '\\':
			dst = append(dst, '\\', b)
			src = src[1:]
		case '\b':
			dst = append(dst, '\\', 'b')
			src = src[1:]
		case '\f':
			dst = append(dst, '\\', 'f')
			src = src[1:]
		case '\n':
			dst = append(dst, '\\', 'n')
			src = src[1:]
		case '\r':
			dst = append(dst, '\\', 'r')
			src = src[1:]
		case '\t':
			dst = append(dst, '\\', 't')
			src = src[1:]
		default:
			if b < 0x20 {
				dst = append(dst, '\\', 'u', '0', '0', hex[b>>4], hex[b&15])
				src = src[1:]
			} else if b < utf8.RuneSelf {
				dst = append(dst, b)
				src = src[1:]
			} else {
				r, size := utf8.DecodeRune(src)
				if r == utf8.RuneError && size == 1 {
					dst = append(dst, 0xef, 0xbf, 0xbd)
					src = src[1:]
					continue
				}
				if r == '\u2028' || r == '\u2029' {
					dst = append(dst, '\\', 'u', '2', '0', '2', hex[byte(r)&15])
				} else {
					dst = append(dst, src[:size]...)
				}
				src = src[size:]
			}
		}
	}
	return append(dst, '"')
}
