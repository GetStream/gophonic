// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

// BenchmarkCPUWorkers exercises the entire warm public transcription path.
func BenchmarkCPUWorkers(b *testing.B) {
	m := loadModel(b, qwen3lm.WeightsF16)
	pcm := clipPCM(b, "jfk")
	for _, workers := range []int{1, 2, 4, 8, 12, 16} {
		b.Run(fmt.Sprint(workers), func(b *testing.B) {
			tr, err := NewTranscriber(m, LaneOptions{Threads: workers})
			if err != nil {
				b.Fatal(err)
			}
			defer tr.Close()
			var dst speech.Transcript
			if err = tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
				b.Fatal(err)
			}
			want := slices.Clone(dst.Text)
			b.ReportAllocs()
			for b.Loop() {
				if err = tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
					b.Fatal(err)
				}
				if !bytes.Equal(dst.Text, want) {
					b.Fatal("transcript changed")
				}
			}
		})
	}
}

// BenchmarkCPUTranscribe checks the complete public call on both languages.
// Its untimed fingerprint lets separately built revisions compare the exact
// encoder output, generated tokens, final hidden state, and final logits.
func BenchmarkCPUTranscribe(b *testing.B) {
	m := loadModel(b, qwen3lm.WeightsF16)
	for _, clip := range []string{"jfk", "zh"} {
		b.Run(clip, func(b *testing.B) {
			tr, err := NewTranscriber(m, LaneOptions{})
			if err != nil {
				b.Fatal(err)
			}
			defer tr.Close()
			pcm := clipPCM(b, clip)
			var dst speech.Transcript
			for range 3 {
				if err = tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
					b.Fatal(err)
				}
			}
			want := slices.Clone(dst.Text)
			h := sha256.New()
			var bits [8]byte
			for _, values := range [][]float32{tr.embeds, tr.hidden, tr.logits} {
				for _, v := range values {
					binary.LittleEndian.PutUint32(bits[:4], math.Float32bits(v))
					h.Write(bits[:4])
				}
			}
			for _, v := range tr.gen {
				binary.LittleEndian.PutUint64(bits[:], uint64(v))
				h.Write(bits[:])
			}
			h.Write(dst.Text)
			b.Logf("state SHA256: %x", h.Sum(nil))
			b.ReportAllocs()
			for b.Loop() {
				if err = tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
					b.Fatal(err)
				}
				if !bytes.Equal(dst.Text, want) {
					b.Fatal("transcript changed")
				}
			}
		})
	}
}
