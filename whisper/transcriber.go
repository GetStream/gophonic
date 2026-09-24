// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"runtime"
)

const WindowSamples = 30 * 16000

var (
	ErrTranscriberClosed       = errors.New("whisper: transcriber is closed")
	ErrTranscriberWindow       = errors.New("whisper: window exceeds 30 seconds of mono 16 kHz PCM")
	ErrTranscriberTextCapacity = errors.New("whisper: transcription output buffer is too small")
)

// Transcriber owns all mutable state for official Whisper tiny.en English
// transcription. A Model can be shared; each concurrent caller needs its own
// Transcriber. Construction allocates scratch and tokenizer tables. Warm calls
// write into caller-owned text buffers without heap allocations.
type Transcriber struct {
	model               *Model
	frontend            *FeatureWorkspace
	fullFrontend        *FullFeatureWorkspace
	encoder             *EncoderWorkspace
	decoder             *DecoderScratch
	tokenizer           *Tokenizer
	policy              *GreedyPolicy
	mel                 []float32
	audio               []float32
	logits              []float32
	tokens              []int
	history             []int
	fullMel             []float32
	segmentText         []byte
	alignTokens         []int
	alignOffsets        []int
	alignProbabilities  []float64
	alignCapture        alignmentCapture
	alignMatrix         []float32
	alignTrace          []byte
	alignCostPrevious   []float64
	alignCostCurrent    []float64
	alignJumps          []int
	alignDurations      []float64
	lastSpeechTimestamp float64
	recordWords         bool
	closed              bool
}

// NewTranscriber prepares one reusable tiny.en worker with at most eight
// execution slots, capped by GOMAXPROCS.
func NewTranscriber(model *Model) (*Transcriber, error) {
	return NewTranscriberWithWorkers(model, min(runtime.GOMAXPROCS(0), 8))
}

// NewTranscriberWithWorkers prepares one reusable tiny.en worker with an
// explicit CPU slot count, including the caller. The count must be 1 through
// 64. Use one Transcriber per concurrent caller and Close it when finished.
func NewTranscriberWithWorkers(model *Model, workers int) (*Transcriber, error) {
	if model == nil {
		return nil, ErrDecoderNilModel
	}
	dims := model.dims
	if dims == (Dims{}) {
		dims = TinyENDims // an unloaded model fails later with a weight error
	}
	encoder, err := newEncoderWorkspace(dims, workers)
	if err != nil {
		return nil, err
	}
	tokenizer, err := NewTokenizer(EnglishOnly)
	if err != nil {
		encoder.Close()
		return nil, err
	}
	policy, err := NewGreedyPolicy(tokenizer, GreedyOptions{WithoutTimestamps: true})
	if err != nil {
		encoder.Close()
		return nil, err
	}
	worker := &Transcriber{
		model:        model,
		frontend:     NewFeatureWorkspace(),
		fullFrontend: NewFullFeatureWorkspace(),
		encoder:      encoder,
		decoder:      newDecoderScratch(dims),
		tokenizer:    tokenizer,
		policy:       policy,
		mel:          make([]float32, MelBins*MelFrames),
		audio:        make([]float32, AudioFrames*dims.AudioState),
		logits:       make([]float32, VocabSize),
		tokens:       make([]int, 0, TextContext+1),
		history:      make([]int, 0, TextContext),
		segmentText:  make([]byte, 0, 4096),
	}
	worker.decoder.gemm = worker.encoder.gemm
	return worker, nil
}

// Close releases this worker's scratch and rejects subsequent use.
func (t *Transcriber) Close() {
	if t == nil || t.closed {
		return
	}
	t.closed = true
	t.frontend.Close()
	t.fullFrontend.Close()
	t.encoder.Close()
	t.decoder = nil
	t.mel = nil
	t.audio = nil
	t.logits = nil
	t.tokens = nil
	t.history = nil
	t.fullMel = nil
	t.segmentText = nil
	t.alignTokens = nil
	t.alignOffsets = nil
	t.alignProbabilities = nil
	t.alignCapture.probabilities = nil
	t.alignCapture.heads = nil
	t.alignMatrix = nil
	t.alignTrace = nil
	t.alignCostPrevious = nil
	t.alignCostCurrent = nil
	t.alignJumps = nil
	t.alignDurations = nil
}

// TranscribeWindowInto appends the transcript of at most 30 seconds of mono
// 16 kHz float32 PCM to dst. It runs the full audio encoder, incremental text
// decoder, official greedy logit policy, and BPE text decoding. The returned
// slice aliases dst; reserve enough capacity before a warm call.
func (t *Transcriber) TranscribeWindowInto(pcm []float32, dst []byte) ([]byte, error) {
	if t == nil || t.closed {
		return dst, ErrTranscriberClosed
	}
	if len(pcm) > WindowSamples {
		return dst, ErrTranscriberWindow
	}
	if err := featuresInto(pcm, t.mel, t.frontend, t.encoder.gemm); err != nil {
		return dst, err
	}
	if err := t.model.EncodeInto(t.mel, t.audio, t.encoder); err != nil {
		return dst, err
	}
	if err := t.model.BeginDecode(t.audio, t.decoder); err != nil {
		return dst, err
	}
	promptLen, err := t.policy.PromptInto(t.tokens[:0], nil, nil)
	if err != nil {
		return dst, err
	}
	t.tokens = t.tokens[:promptLen]
	for position, tokenID := range t.tokens {
		if err := t.model.LogitsForTokenInto(tokenID, position, t.decoder, t.logits); err != nil {
			return dst, err
		}
	}
	textEnd := promptLen
	for generated := 0; generated < TextContext/2 && len(t.tokens) <= TextContext; generated++ {
		next, err := t.policy.SelectNextInto(t.logits, t.logits, t.tokens)
		if err != nil {
			return dst, err
		}
		t.tokens = append(t.tokens, next)
		if next == t.tokenizer.EOT() {
			break
		}
		textEnd = len(t.tokens)
		if generated+1 >= TextContext/2 || len(t.tokens) > TextContext {
			break
		}
		if err := t.model.LogitsForTokenInto(next, len(t.tokens)-1, t.decoder, t.logits); err != nil {
			return dst, err
		}
	}
	start := len(dst)
	decoded, err := t.tokenizer.DecodeInto(dst, t.tokens[promptLen:textEnd])
	if err != nil {
		return dst, err
	}
	trimmed := trimWhisperSpace(decoded[start:])
	copy(decoded[start:], trimmed)
	return t.repairTranscriptionUTF8(decoded[:start+len(trimmed)], start)
}

// TranscribeFixedWindowsInto handles any length of mono 16 kHz PCM as
// consecutive 30-second windows, appending text to dst. It is a simple
// fixed-window mode: timestamp-based seeking, silence skipping, and previous
// window prompting are not applied.
func (t *Transcriber) TranscribeFixedWindowsInto(pcm []float32, dst []byte) ([]byte, error) {
	if t == nil || t.closed {
		return dst, ErrTranscriberClosed
	}
	if len(pcm) <= WindowSamples {
		return t.TranscribeWindowInto(pcm, dst)
	}
	for offset := 0; offset < len(pcm); offset += WindowSamples {
		end := min(offset+WindowSamples, len(pcm))
		before := len(dst)
		if before != 0 {
			if len(dst) == cap(dst) {
				return dst, ErrTranscriberTextCapacity
			}
			dst = append(dst, ' ')
		}
		var err error
		dst, err = t.TranscribeWindowInto(pcm[offset:end], dst)
		if err != nil {
			return dst, err
		}
		if len(dst) == before+1 && before != 0 {
			dst = dst[:before]
		}
	}
	return dst, nil
}
