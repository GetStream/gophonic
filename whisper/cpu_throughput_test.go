// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin || linux

package whisper

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/cputest"
	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
)

func TestCPUConcurrentThroughput(t *testing.T) {
	if os.Getenv("STT_SERVER_OUTPUT") == "" {
		t.Skip("set STT_SERVER_OUTPUT")
	}
	m, err := Load(testmodels.Path(t, testmodels.WhisperTinyEN))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	data, err := os.ReadFile("../testdata/whisper_jfk.pcm.f32le")
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	cputest.Run(t, func(index, workers int) cputest.Lane {
		tr, err := NewTranscriber(m, LaneOptions{Threads: workers})
		if err != nil {
			t.Fatal(err)
		}
		input := pcm
		if index%2 == 1 {
			input = input[:len(input)*3/4]
		}
		var out speech.Transcript
		return cputest.Lane{
			Run: func() error { return tr.Transcribe(context.Background(), input, speech.Options{}, &out) },
			Fingerprint: func() [32]byte {
				h := sha256.New()
				var bits [8]byte
				for _, values := range [][]float32{tr.audio, tr.logits} {
					for _, v := range values {
						binary.LittleEndian.PutUint32(bits[:4], math.Float32bits(v))
						h.Write(bits[:4])
					}
				}
				for _, id := range tr.tokens {
					binary.LittleEndian.PutUint64(bits[:], uint64(id))
					h.Write(bits[:])
				}
				h.Write(out.Text)
				h.Write([]byte{byte(out.Language)})
				var sum [32]byte
				h.Sum(sum[:0])
				return sum
			},
			Close: func() { tr.Close() },
		}
	})
}
