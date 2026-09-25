// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

// BenchmarkCPULogitsRows projects real prompt-tail hidden states through the
// vocabulary head. Separate calls are the exact reference for the CPU batch.
func BenchmarkCPULogitsRows(b *testing.B) {
	m := loadModel(b, FormatF16)
	tr, err := NewTranscriber(m, 0)
	if err != nil {
		b.Fatal(err)
	}
	defer tr.Close()
	var transcript speech.Transcript
	if err := tr.Transcribe(context.Background(), clipPCM(b, "jfk"), speech.Options{}, &transcript); err != nil {
		b.Fatal(err)
	}
	c := m.lm.Config()
	hidden := make([]float32, 32*c.Hidden)
	kv, err := m.eval.NewPrefixKV(len(tr.ids))
	if err != nil {
		b.Fatal(err)
	}
	defer kv.Close()
	if err := m.eval.HiddenTailExtendEmbedInto(kv, 0, tr.ids, qwen3lm.Embeds{Token: m.ids.audioPad, Rows: tr.embeds}, hidden, tr.lm); err != nil {
		b.Fatal(err)
	}
	for _, rows := range []int{1, 3, 8, 16, 32} {
		b.Run(fmt.Sprint(rows), func(b *testing.B) {
			src := hidden[:rows*c.Hidden]
			got, want := make([]float32, rows*c.Vocab), make([]float32, rows*c.Vocab)
			separate := func(dst []float32) {
				for r := range rows {
					if err := m.eval.LogitsInto(src[r*c.Hidden:(r+1)*c.Hidden], dst[r*c.Vocab:(r+1)*c.Vocab], tr.lm); err != nil {
						b.Fatal(err)
					}
				}
			}
			separate(want)
			if err := m.eval.LogitsRowsInto(src, got, tr.lm); err != nil {
				b.Fatal(err)
			}
			for i, v := range got {
				if math.Float32bits(v) != math.Float32bits(want[i]) {
					b.Fatalf("logit %d differs from separate calls", i)
				}
			}
			b.Run("separate", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					separate(got)
				}
			})
			b.Run("batch", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := m.eval.LogitsRowsInto(src, got, tr.lm); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

// BenchmarkCPUPartialTranscribe measures four public calls on strictly
// growing PCM. Each continuation verifies the preceding call's transcript.
func BenchmarkCPUPartialTranscribe(b *testing.B) {
	m := loadModel(b, FormatF16)
	for _, clip := range []string{"jfk", "zh"} {
		b.Run(clip, func(b *testing.B) {
			tr, err := NewTranscriber(m, 0)
			if err != nil {
				b.Fatal(err)
			}
			defer tr.Close()
			pcm := clipPCM(b, clip)
			var prev, cur speech.Transcript
			var expected [4][]byte
			run := func(first bool) {
				prev.Reset()
				cur.Reset()
				for step := range 4 {
					opts := speech.Options{}
					if step > 0 {
						opts.Partial = &prev
					}
					if err := tr.Transcribe(context.Background(), pcm[:len(pcm)*(step+1)/4], opts, &cur); err != nil {
						b.Fatal(err)
					}
					if first {
						expected[step] = slices.Clone(cur.Text)
					} else if !bytes.Equal(cur.Text, expected[step]) {
						b.Fatalf("step %d text changed", step)
					}
					prev, cur = cur, prev
				}
			}
			run(true)
			run(false)
			b.ReportAllocs()
			for b.Loop() {
				run(false)
			}
		})
	}
}
