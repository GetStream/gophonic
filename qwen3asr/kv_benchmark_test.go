// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

// BenchmarkDecoderPrefixLength uses the real F16 model and its audio prompt,
// extended by valid repeated text tokens to isolate context-length scaling.
// Prefill is untimed; every iteration performs one real cached token and head.
func BenchmarkDecoderPrefixLength(b *testing.B) {
	m := loadModel(b, qwen3lm.WeightsF16)
	tr, err := NewTranscriber(m, LaneOptions{})
	if err != nil {
		b.Fatal(err)
	}
	defer tr.Close()
	var transcript speech.Transcript
	if err := tr.Transcribe(context.Background(), clipPCM(b, "jfk"), speech.Options{}, &transcript); err != nil {
		b.Fatal(err)
	}
	for _, length := range []int{256, 512, 2048} {
		b.Run(fmt.Sprint(length), func(b *testing.B) {
			ids := make([]int, length)
			copy(ids, tr.ids)
			for i := len(tr.ids); i < len(ids); i++ {
				ids[i] = tr.gen[0]
			}
			kv, err := m.eval.NewPrefixKV(length + 1)
			if err != nil {
				b.Fatal(err)
			}
			defer kv.Close()
			embeds := qwen3lm.Embeds{Token: m.ids.audioPad, Rows: tr.embeds}
			if err := m.eval.HiddenLastExtendEmbedInto(kv, 0, ids, embeds, tr.hidden, tr.lm); err != nil {
				b.Fatal(err)
			}
			step := tr.gen[:1]
			if err := m.eval.HiddenLastExtendInto(kv, length, step, tr.hidden, tr.lm); err != nil {
				b.Fatal(err)
			}
			if err := m.eval.LogitsInto(tr.hidden, tr.logits, tr.lm); err != nil {
				b.Fatal(err)
			}
			want := tr.logits[step[0]]
			b.ReportAllocs()
			for b.Loop() {
				if err := m.eval.HiddenLastExtendInto(kv, length, step, tr.hidden, tr.lm); err != nil {
					b.Fatal(err)
				}
				if err := m.eval.LogitsInto(tr.hidden, tr.logits, tr.lm); err != nil {
					b.Fatal(err)
				}
				if math.IsNaN(float64(tr.logits[step[0]])) || tr.logits[step[0]] != want {
					b.Fatal("unstable cached token")
				}
			}
		})
	}
}
