// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin || linux

package qwen3asr

import (
	"context"
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/cputest"
	"github.com/GetStream/gophonic/speech"
)

func TestCPUConcurrentThroughput(t *testing.T) {
	if os.Getenv("STT_SERVER_OUTPUT") == "" {
		t.Skip("set STT_SERVER_OUTPUT")
	}
	m := loadModel(t, FormatF16)
	clip := "zh"
	if os.Getenv("STT_SERVER_CLIP") == "jfk" {
		clip = "jfk"
	}
	pcm := clipPCM(t, clip)
	cputest.Run(t, func(index, workers int) cputest.Lane {
		tr, err := NewTranscriber(m, workers)
		if err != nil {
			t.Fatal(err)
		}
		input := pcm
		if index%2 == 1 {
			input = input[:len(input)*3/4]
		}
		var out speech.Transcript
		return cputest.Lane{
			Run:         func() error { return tr.Transcribe(context.Background(), input, speech.Options{}, &out) },
			Fingerprint: func() [32]byte { return continuationFingerprint(tr, &out) },
			Close:       func() { tr.Close() },
		}
	})
}
