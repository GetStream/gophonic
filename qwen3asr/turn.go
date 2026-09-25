// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"math"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/internal/vec"
	"github.com/GetStream/gophonic/speech"
)

// The decoder's state as it ends a transcript has heard the audio and read
// the words, so it knows whether the speaker sounds done and whether what
// they said is. A logistic regression over that state judges the end of a
// turn; tools/turn trains it on labeled human and synthetic speech cut at
// pauses (pipecat-ai/smart-turn-data-v3.2-train) and scores it on held-out
// human speech. The file holds a header, the geometry of the checkpoint it
// was trained on, the decision threshold, the bias, and one weight per
// state value, in little-endian binary.
//
//go:embed turn-1.7b.bin
var turn17 []byte

const turnMagic = "gophonic turn 1\n"

// turnHead judges the end of a turn from the state that ends a transcript.
type turnHead struct {
	weights   []float32
	bias      float32
	threshold float32
}

// turnHeadFor returns the head trained on the checkpoint geometry of lm, or
// nil if none was.
func turnHeadFor(lm qwen3lm.Config) *turnHead {
	h, err := parseTurnHead(turn17)
	if err != nil {
		panic(err) // the embedded file is part of the package
	}
	if h.hidden != lm.Hidden || h.layers != lm.Layers || h.vocab != lm.Vocab {
		return nil
	}
	return &h.turnHead
}

type turnFile struct {
	turnHead
	hidden, layers, vocab int
}

func parseTurnHead(raw []byte) (turnFile, error) {
	var f turnFile
	const header = len(turnMagic) + 5*4
	if len(raw) < header || string(raw[:len(turnMagic)]) != turnMagic {
		return f, errors.New("qwen3asr: malformed turn head")
	}
	u := func(i int) uint32 { return binary.LittleEndian.Uint32(raw[len(turnMagic)+4*i:]) }
	f.hidden, f.layers, f.vocab = int(u(0)), int(u(1)), int(u(2))
	f.threshold, f.bias = math.Float32frombits(u(3)), math.Float32frombits(u(4))
	if len(raw) != header+4*f.hidden {
		return f, errors.New("qwen3asr: malformed turn head")
	}
	f.weights = make([]float32, f.hidden)
	for i := range f.weights {
		f.weights[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[header+4*i:]))
	}
	return f, nil
}

// predict judges the turn from the state that ended a transcript.
func (h *turnHead) predict(state []float32) speech.Prediction {
	p := float32(1 / (1 + math.Exp(-float64(vec.Dot(h.weights, state)+h.bias))))
	return speech.Prediction{Probability: p, Complete: p >= h.threshold}
}
