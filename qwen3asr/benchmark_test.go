// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
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

// BenchmarkEncoder measures the audio encoder alone on the JFK clip's
// features, on the CPU (FormatF16 models) and the GPU (FormatGPU models).
func BenchmarkEncoder(b *testing.B) {
	for _, format := range []string{FormatF16, FormatGPU} {
		b.Run(format, func(b *testing.B) {
			m := loadModel(b, format)
			tr, err := NewTranscriber(m, 0)
			if err != nil {
				b.Fatal(err)
			}
			defer tr.Close()
			pcm := clipPCM(b, "jfk")
			var dst speech.Transcript
			if err := tr.Transcribe(context.Background(), pcm, speech.Options{}, &dst); err != nil {
				b.Fatal(err)
			}
			frames := len(tr.features) / m.enc.freq[0]
			b.ReportAllocs()
			for b.Loop() {
				if err := tr.encode(frames); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkDecoder measures the decoder on the JFK clip's prompt: a fresh
// prefill of all 158 tokens, and one decoding step with its logits.
func BenchmarkDecoder(b *testing.B) {
	for _, format := range []string{FormatF16, FormatGPU} {
		m := loadModel(b, format)
		tr, err := NewTranscriber(m, 0)
		if err != nil {
			b.Fatal(err)
		}
		var dst speech.Transcript
		if err := tr.Transcribe(context.Background(), clipPCM(b, "jfk"), speech.Options{}, &dst); err != nil {
			b.Fatal(err)
		}
		embeds := qwen3lm.Embeds{Token: m.ids.audioPad, Rows: tr.embeds}
		kv, err := m.eval.NewPrefixKV(len(tr.ids) + 64)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(format+"/prefill", func(b *testing.B) {
			for b.Loop() {
				if err := m.eval.HiddenLastExtendEmbedInto(kv, 0, tr.ids, embeds, tr.hidden, tr.lm); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(format+"/token", func(b *testing.B) {
			if err := m.eval.HiddenLastExtendEmbedInto(kv, 0, tr.ids, embeds, tr.hidden, tr.lm); err != nil {
				b.Fatal(err)
			}
			step := tr.gen[:1]
			for b.Loop() {
				if err := m.eval.HiddenLastExtendInto(kv, len(tr.ids), step, tr.hidden, tr.lm); err != nil {
					b.Fatal(err)
				}
				if err := m.eval.LogitsInto(tr.hidden, tr.logits, tr.lm); err != nil {
					b.Fatal(err)
				}
			}
		})
		tr.Close()
	}
}
