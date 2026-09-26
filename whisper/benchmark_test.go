// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

var whisperBenchmarkSink float32

func BenchmarkOfficialTinyENEncoder(b *testing.B) {
	path := testmodels.Path(b, testmodels.WhisperTinyEN)
	m, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	mel, err := readFloatFixture(filepath.Join("..", "testdata", "whisper", "jfk.mel.f32le"), MelBins*MelFrames)
	if err != nil {
		b.Fatal(err)
	}
	w := newTinyEncoderWorkspace()
	defer w.Close()
	out := make([]float32, audioFrames*audioState)
	if err := m.encode(mel, out, w); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := m.encode(mel, out, w); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	whisperBenchmarkSink = out[0]
}

// BenchmarkOfficialTinyENTranscribe measures the full warm mono-PCM to text
// path: whole-file log-mel, encoder, incremental decoder, and BPE. The model
// load, transcriber construction, file read, and first weight packing are
// outside the timed loop.
func BenchmarkOfficialTinyENTranscribe(b *testing.B) {
	path := testmodels.Path(b, testmodels.WhisperTinyEN)
	m, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	worker, err := NewTranscriber(m, LaneOptions{})
	if err != nil {
		b.Fatal(err)
	}
	defer worker.Close()
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		b.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	text := make([]byte, 0, 2048)
	if _, err := worker.TranscribeInto(pcm, text); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := worker.TranscribeInto(pcm, text); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOfficialTinyENWindow measures the 30-second PCM-padded decode used
// by the pinned single-window JFK oracle. Compare it with CPU references only
// when the input padding and resulting token sequence match.
func BenchmarkOfficialTinyENWindow(b *testing.B) {
	path := testmodels.Path(b, testmodels.WhisperTinyEN)
	m, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	worker, err := NewTranscriber(m, LaneOptions{})
	if err != nil {
		b.Fatal(err)
	}
	defer worker.Close()
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		b.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	text := make([]byte, 0, 2048)
	if _, err := worker.TranscribeWindowInto(pcm, text); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := worker.TranscribeWindowInto(pcm, text); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOfficialTinyENTimestamps isolates the optional segment and word
// timing paths on the same pinned JFK workload as the plain-text benchmark.
func BenchmarkOfficialTinyENTimestamps(b *testing.B) {
	path := testmodels.Path(b, testmodels.WhisperTinyEN)
	model, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		b.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	for _, mode := range []string{"segments", "words"} {
		b.Run(mode, func(b *testing.B) {
			worker, err := NewTranscriber(model, LaneOptions{})
			if err != nil {
				b.Fatal(err)
			}
			defer worker.Close()
			text := make([]byte, 0, 4096)
			segments := make([]Segment, 0, 64)
			words := make([]Word, 0, 128)
			run := func() error {
				if mode == "words" {
					_, _, _, err := worker.TranscribeWordsInto(pcm, text[:0], segments[:0], words[:0])
					return err
				}
				_, _, err := worker.TranscribeSegmentsInto(pcm, text[:0], segments[:0])
				return err
			}
			if err := run(); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err := run(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
