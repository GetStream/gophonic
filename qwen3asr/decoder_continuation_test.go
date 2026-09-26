// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

// Cancellation is observed at transcribe's check after encoding and prompt
// construction, before the decoder has consumed the new embeddings.
type cancelAfterEncode struct {
	context.Context
	checks int
}

func (c *cancelAfterEncode) Err() error {
	c.checks++
	if c.checks >= 2 {
		return context.Canceled
	}
	return nil
}

func continuationFingerprint(tr *Transcriber, out *speech.Transcript) [32]byte {
	h := sha256.New()
	var bits [8]byte
	add := func(values []float32) {
		for _, v := range values {
			binary.LittleEndian.PutUint32(bits[:4], math.Float32bits(v))
			h.Write(bits[:4])
		}
	}
	add(tr.embeds)
	add(tr.hidden)
	add(tr.logits)
	if len(tr.draft) > 0 {
		add(tr.tail)
		add(tr.vlogits[:cap(tr.vlogits)])
	}
	for _, id := range tr.gen {
		binary.LittleEndian.PutUint64(bits[:], uint64(id))
		h.Write(bits[:])
	}
	h.Write(out.Text)
	h.Write([]byte{byte(out.Language)})
	var outHash [32]byte
	h.Sum(outHash[:0])
	return outHash
}

func TestDecoderContinuationGrowingPCMExact(t *testing.T) {
	m := loadModel(t, qwen3lm.WeightsF16)
	tr, err := NewTranscriber(m, LaneOptions{Threads: 8})
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	step := m.enc.prefixStep()
	frames := []int{step + 2, step + 100, step + 200, step + 400, step + 500}
	clip := clipPCM(t, "jfk")
	clip = clip[:len(clip)/160*160]
	pcm := make([]float32, frames[len(frames)-1]*160)
	for i := range pcm {
		pcm[i] = clip[i%len(clip)]
	}
	changed := slices.Clone(pcm)
	// Alter the beginning before the abandoned encode, so reusing KV from
	// the last successful decoder would be observably wrong on the next call.
	for i := range min(len(changed), 4*sampleRate) {
		changed[i] *= 0.5
	}
	var prev, cur speech.Transcript
	var want [5][32]byte
	run := func(reuse, capture bool) {
		prev.Reset()
		cur.Reset()
		var used [5]int
		for i, f := range frames {
			if !reuse {
				tr.decoderPrefix.reset()
			}
			if i == 3 {
				if reuse && !tr.decoderPrefix.valid {
					t.Fatal("abandoned encode did not start with a valid decoder prefix")
				}
				revision := tr.audioRevision
				ctx := &cancelAfterEncode{Context: context.Background()}
				err := tr.Transcribe(ctx, changed[:(step+300)*160], speech.Options{Partial: &prev}, &cur)
				if !errors.Is(err, context.Canceled) || tr.audioRevision != revision+1 {
					t.Fatalf("encode was not abandoned at the expected boundary: revision=%d err=%v", tr.audioRevision, err)
				}
			}
			before := tr.decoderPrefix
			clear(tr.vlogits[:cap(tr.vlogits)])
			input := pcm
			if i >= 3 {
				input = changed
			}
			opts := speech.Options{Partial: &prev}
			if i == 4 {
				opts.Context = "Names mentioned in this recording: Lunar."
			}
			if err := tr.Transcribe(context.Background(), input[:f*160], opts, &cur); err != nil {
				t.Fatal(err)
			}
			used[i] = before.reusable(tr.decoderPrefix.anchor, tr.reusedAudioRows, tr.audioRevision) / 128 * 128
			if i == 3 && tr.reusedAudioRows == 0 {
				t.Fatal("abandoned-encode check did not reach the encoder prefix reuse path")
			}
			if capture {
				want[i] = continuationFingerprint(tr, &cur)
			} else if reuse && continuationFingerprint(tr, &cur) != want[i] {
				t.Fatalf("call%d cached continuation changed a float bit, token, text or language", i)
			}
			prev, cur = cur, prev
		}
		// The preceding trace ends with another context, so the first call
		// may rebuild the static prompt before the ordinary audio anchor is
		// established. The third call must still reach positive KV reuse.
		if reuse && (used[2] == 0 || used[3] != 0 || used[4] != 0) {
			t.Fatalf("reuse/invalidation paths not exercised: %v", used)
		}
	}
	run(false, false) // Warm capacities and the static prompt's arithmetic history.
	run(false, true)
	run(true, false)
}
