// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package audiofile decodes the file formats accepted by the gophonic CLI and server.
package audiofile

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Decode returns interleaved float32 PCM, sample rate, and channel count.
// maxSeconds limits decoded audio duration when positive. Callers handling
// untrusted input should separately limit encoded input bytes.
func Decode(name string, input io.Reader, maxSeconds int) ([]float32, int, int, error) {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".wav":
		data, err := io.ReadAll(input)
		if err != nil {
			return nil, 0, 0, err
		}
		return decodeWAV(data, maxSeconds)
	case ".opus", ".ogg":
		return decodeOggOpus(input, maxSeconds)
	default:
		return nil, 0, 0, fmt.Errorf("unsupported audio file %q; use WAV or Ogg Opus", name)
	}
}

// DecodeBytes decodes an in-memory upload without copying a WAV payload.
func DecodeBytes(name string, data []byte, maxSeconds int) ([]float32, int, int, error) {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".wav":
		return decodeWAV(data, maxSeconds)
	case ".opus", ".ogg":
		return decodeOggOpus(bytes.NewReader(data), maxSeconds)
	default:
		return nil, 0, 0, fmt.Errorf("unsupported audio file %q; use WAV or Ogg Opus", name)
	}
}

// ReadPath decodes a local file without a duration limit.
func ReadPath(path string) ([]float32, int, int, error) {
	input, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer input.Close()
	return Decode(path, input, 0)
}
