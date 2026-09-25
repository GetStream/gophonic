// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"bytes"
	"context"
	"fmt"
	"github.com/GetStream/gophonic/speech"
	"slices"
	"testing"
)

// BenchmarkCPUWorkers exercises the entire warm public transcription path.
func BenchmarkCPUWorkers(b *testing.B) {
	m := loadModel(b, FormatF16)
	pcm := clipPCM(b, "jfk")
	for _, workers := range []int{1, 2, 4, 8, 12, 16} {
		b.Run(fmt.Sprint(workers), func(b *testing.B) {
			tr, err := NewTranscriber(m, workers)
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
