// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"bytes"
	"runtime"
	"unsafe"

	"github.com/GetStream/gophonic/internal/arena"
	"github.com/GetStream/gophonic/internal/q8gemm"
)

// The normalized feature snapshot is small relative to encoder activations,
// pointer-free, lane-owned, and bounded even for the longest accepted audio.
const encoderPrefixBytes = 8 << 20

type encoderPrefix struct {
	memory             *arena.Arena
	features           []float32
	frames, seenFrames int
}

func (p *encoderPrefix) reset() { p.frames, p.seenFrames = 0, 0 }

func (p *encoderPrefix) close() {
	_ = p.memory.Close()
	*p = encoderPrefix{}
}

// prefixStep requires complete attention windows and complete activation
// tiles. The latter preserves the original row-versus-tile dispatch at the
// end of the suffix, whose FP32 reduction orders are different.
func (e *encoder) prefixStep() int {
	tokens := frameTokens(e.chunkFrames) * e.windowChunks
	windows := 1
	for tokens*windows%q8gemm.ActivationRows != 0 {
		windows++
	}
	return e.chunkFrames * e.windowChunks * windows
}

func (p *encoderPrefix) reusable(e *encoder, features []float32, frames int) int {
	defer runtime.KeepAlive(p)
	if p.frames == 0 || frames <= p.seenFrames || p.frames >= frames {
		return 0
	}
	for band := range e.freq[0] {
		old := p.features[band*p.frames : (band+1)*p.frames]
		now := features[band*frames : band*frames+p.frames]
		if !bytes.Equal(featureBits(old), featureBits(now)) {
			return 0
		}
	}
	return p.frames
}

// featureBits exposes only the numeric slice's existing bytes. It neither
// allocates nor extends its bounds, and callers retain the native owner.
func featureBits(values []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), 4*len(values))
}

func (p *encoderPrefix) remember(e *encoder, features []float32, frames int, unchanged bool) {
	defer runtime.KeepAlive(p)
	step := e.prefixStep()
	// Leave the last two STFT frames uncached: right-edge reflection can
	// change them when more PCM arrives, even without a new global peak.
	complete := min(max(0, frames-2), encoderPrefixBytes/(4*e.freq[0])) / step * step
	p.seenFrames = frames
	if unchanged && complete == p.frames {
		return
	}
	p.frames = 0
	if complete == 0 {
		return
	}
	n := complete * e.freq[0]
	if cap(p.features) < n {
		capacity := min(encoderPrefixBytes/4, max(n, 2*cap(p.features)))
		memory, err := arena.New(capacity)
		if err != nil {
			return // caching is optional; the completed encode remains valid
		}
		data := memory.Take(capacity)
		_ = p.memory.Close()
		p.memory, p.features = memory, data
	}
	p.features = p.features[:n]
	for band := range e.freq[0] {
		copy(p.features[band*complete:(band+1)*complete], features[band*frames:band*frames+complete])
	}
	p.frames = complete
}
