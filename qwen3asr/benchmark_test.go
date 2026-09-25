// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"testing"

	"github.com/GetStream/gophonic/speech"
)

// BenchmarkTranscribe measures warm transcription of the test clips, PCM to
// text, and reports the real-time factor.
func BenchmarkTranscribe(b *testing.B) {
	for _, format := range []string{FormatF16, FormatGPU} {
		for _, clip := range []string{"jfk", "zh"} {
			b.Run(format+"/"+clip, func(b *testing.B) {
				tr, err := NewTranscriber(loadModel(b, format), 0)
				if err != nil {
					b.Fatal(err)
				}
				defer tr.Close()
				pcm := clipPCM(b, clip)
				var dst speech.Transcript
				if err := tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				for b.Loop() {
					if err := tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(b.Elapsed().Seconds())/float64(b.N)/(float64(len(pcm))/sampleRate), "RTF")
			})
		}
	}
}
