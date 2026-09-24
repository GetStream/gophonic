// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package httpserver

import (
	"bytes"
	"errors"
	"strings"
)

var (
	errMultipart = errors.New("expected multipart/form-data with a boundary")
	errFile      = errors.New("file is required")
	errField     = errors.New("unsupported form field")
	errValue     = errors.New("unsupported form value")
)

// upload aliases the caller's bounded request body. No part bytes are copied.
type upload struct {
	name  []byte
	audio []byte
	text  bool
}

func parseUpload(contentType string, body []byte) (upload, error) {
	const prefix = "multipart/form-data;"
	if !strings.HasPrefix(contentType, prefix) {
		return upload{}, errMultipart
	}
	params := strings.TrimSpace(contentType[len(prefix):])
	if !strings.HasPrefix(params, "boundary=") {
		return upload{}, errMultipart
	}
	boundary := strings.TrimSpace(params[len("boundary="):])
	if len(boundary) >= 2 && boundary[0] == '"' && boundary[len(boundary)-1] == '"' {
		boundary = boundary[1 : len(boundary)-1]
	}
	if len(boundary) == 0 || len(boundary) > 70 || strings.ContainsAny(boundary, "\r\n") {
		return upload{}, errMultipart
	}
	var markerBuf [72]byte
	markerBuf[0], markerBuf[1] = '-', '-'
	marker := markerBuf[:2+copy(markerBuf[2:], boundary)]
	var nextMarkerBuf [74]byte
	nextMarkerBuf[0], nextMarkerBuf[1] = '\r', '\n'
	nextMarker := nextMarkerBuf[:2+copy(nextMarkerBuf[2:], marker)]
	if !bytes.HasPrefix(body, marker) {
		return upload{}, errMultipart
	}
	var result upload
	for pos := len(marker); ; {
		if len(body[pos:]) >= 2 && bytes.Equal(body[pos:pos+2], []byte("--")) {
			if len(result.audio) == 0 {
				return upload{}, errFile
			}
			return result, nil
		}
		if len(body[pos:]) < 2 || !bytes.Equal(body[pos:pos+2], []byte("\r\n")) {
			return upload{}, errMultipart
		}
		pos += 2
		headersEnd := bytes.Index(body[pos:], []byte("\r\n\r\n"))
		if headersEnd < 0 || headersEnd > 8192 {
			return upload{}, errMultipart
		}
		header := body[pos : pos+headersEnd]
		field, filename, err := parseDisposition(header)
		if err != nil {
			return upload{}, err
		}
		pos += headersEnd + 4
		next := 0
		for {
			match := bytes.Index(body[pos+next:], nextMarker)
			if match < 0 {
				return upload{}, errMultipart
			}
			next += match
			after := pos + next + len(nextMarker)
			if after+2 <= len(body) && (bytes.Equal(body[after:after+2], []byte("--")) || bytes.Equal(body[after:after+2], []byte("\r\n"))) {
				break
			}
			next++
		}
		value := body[pos : pos+next]
		pos += next + 2 + len(marker)
		switch {
		case bytes.Equal(field, []byte("file")):
			if result.audio != nil || len(filename) == 0 {
				return upload{}, errFile
			}
			result.name, result.audio = filename, value
		case bytes.Equal(field, []byte("model")):
			if len(value) > 128 || len(value) != 0 && !bytes.Equal(value, []byte("gophonic-whisper")) && !bytes.Equal(value, []byte("whisper-1")) {
				return upload{}, errValue
			}
		case bytes.Equal(field, []byte("response_format")):
			if bytes.Equal(value, []byte("text")) {
				result.text = true
			} else if !bytes.Equal(value, []byte("json")) {
				return upload{}, errValue
			}
		case bytes.Equal(field, []byte("language")):
			if len(value) != 0 && !bytes.Equal(value, []byte("en")) {
				return upload{}, errValue
			}
		default:
			return upload{}, errField
		}
	}
}

func parseDisposition(header []byte) (field, filename []byte, err error) {
	for len(header) > 0 {
		end := bytes.Index(header, []byte("\r\n"))
		if end < 0 {
			end = len(header)
		}
		line := header[:end]
		if bytes.HasPrefix(line, []byte("Content-Disposition: form-data;")) {
			args := line[len("Content-Disposition: form-data;"):]
			for len(args) > 0 {
				args = bytes.TrimSpace(args)
				if len(args) == 0 {
					break
				}
				cut := bytes.IndexByte(args, ';')
				var item []byte
				if cut < 0 {
					item, args = args, nil
				} else {
					item, args = args[:cut], args[cut+1:]
				}
				item = bytes.TrimSpace(item)
				if bytes.HasPrefix(item, []byte("name=\"")) && len(item) >= 7 && item[len(item)-1] == '"' {
					field = item[6 : len(item)-1]
				}
				if bytes.HasPrefix(item, []byte("filename=\"")) && len(item) >= 11 && item[len(item)-1] == '"' {
					filename = item[10 : len(item)-1]
				}
			}
		}
		if end == len(header) {
			break
		}
		header = header[end+2:]
	}
	if len(field) == 0 {
		return nil, nil, errMultipart
	}
	return field, filename, nil
}
