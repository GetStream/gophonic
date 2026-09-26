// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"errors"
	"testing"

	"github.com/GetStream/gophonic/internal/arena"
	"github.com/GetStream/gophonic/internal/mel"
	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

func TestTranscriberCloseRetiresStorage(t *testing.T) {
	memory, err := arena.New(8, 16)
	if err != nil {
		t.Fatal(err)
	}
	defer memory.Close()
	tr := &Transcriber{m: &Model{}, frontend: mel.NewSpectrogram(80, 0),
		decodeMemory: memory, hidden: memory.Take(8), logits: memory.Take(16),
		pcm: make([]float32, 1024), vlogits: make([]float32, 1024),
		lm: &qwen3lm.Workspace{}}
	if err := tr.Close(); err != nil {
		t.Fatal(err)
	}
	if memory.Bytes() != 0 {
		t.Fatal("close retained the native numeric arena")
	}
	if tr.m != nil || tr.lm != nil || tr.frontend != nil || tr.pcm != nil || tr.vlogits != nil || tr.hidden != nil || tr.logits != nil {
		t.Fatal("closed transcriber retains model or high-water storage")
	}
	if err := tr.Close(); err != nil {
		t.Fatal(err)
	}
	if err := tr.Transcribe(context.Background(), nil, speech.Options{}, nil); !errors.Is(err, speech.ErrClosed) {
		t.Fatalf("closed transcriber: %v", err)
	}
}

func TestModelCloseRetiresStorage(t *testing.T) {
	memory, err := arena.New(16)
	if err != nil {
		t.Fatal(err)
	}
	defer memory.Close()
	m := &Model{enc: &encoder{memory: memory}, lm: &qwen3lm.Weights{}, languages: speech.Languages(speech.English)}
	m.Close()
	if memory.Bytes() != 0 || m.enc != nil || m.lm != nil || m.languages != 0 {
		t.Fatal("released model retains weights or native storage")
	}
	m.Close()
	(*Model)(nil).Close()
	if _, err := NewTranscriber(m, LaneOptions{Threads: 1}); err == nil {
		t.Fatal("released model opened a transcriber")
	}
	if _, err := NewBatchTranscriber(m, 2, LaneOptions{Threads: 1}); err == nil {
		t.Fatal("released model opened a batch transcriber")
	}
}
