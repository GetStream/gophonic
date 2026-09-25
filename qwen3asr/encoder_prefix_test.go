// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"math"
	"testing"
)

func TestEncoderPrefixInvalidation(t *testing.T) {
	e := &encoder{chunkFrames: 100, windowChunks: 8, freq: [4]int{2}}
	if n := e.prefixStep(); n != 1600 {
		t.Fatalf("aligned window step=%d", n)
	}
	var p encoderPrefix
	defer p.close()
	makeFeatures := func(frames int) []float32 {
		values := make([]float32, frames*2)
		for band := range 2 {
			for i := range frames {
				values[band*frames+i] = float32((band*13+i)%31) / 16
			}
		}
		return values
	}
	p.remember(e, makeFeatures(1602), 1602, false)
	next := makeFeatures(1650)
	if got := p.reusable(e, next, 1650); got != 1600 {
		t.Fatalf("growing unchanged features reuse=%d", got)
	}
	if got := p.reusable(e, makeFeatures(1602), 1602); got != 0 {
		t.Fatal("reused a repeated input instead of a growing continuation")
	}
	next[0] = math.Float32frombits(0x80000000)
	if got := p.reusable(e, next, 1650); got != 0 {
		t.Fatal("ignored a signed-zero bit change")
	}
	next[0] = 0
	next[1650+100]++
	if got := p.reusable(e, next, 1650); got != 0 {
		t.Fatal("ignored a changed earlier feature")
	}
	p.reset()
	if got := p.reusable(e, makeFeatures(1650), 1650); got != 0 {
		t.Fatal("reused an invalidated prefix")
	}
}

func TestEncoderPrefixBound(t *testing.T) {
	e := &encoder{chunkFrames: 100, windowChunks: 8, freq: [4]int{128}}
	var p encoderPrefix
	defer p.close()
	frames := 2*encoderPrefixBytes/(4*e.freq[0]) + 7
	features := make([]float32, frames*e.freq[0])
	p.remember(e, features, frames, false)
	want := encoderPrefixBytes / (4 * e.freq[0]) / e.prefixStep() * e.prefixStep()
	if p.frames != want || p.memory.Bytes() > encoderPrefixBytes {
		t.Fatalf("cached frames=%d, want=%d, bytes=%d", p.frames, want, p.memory.Bytes())
	}
	p.reset()
	if p.frames != 0 || p.seenFrames != 0 {
		t.Fatal("reset retained a valid prefix")
	}
	if p.memory.Bytes() == 0 {
		t.Fatal("reset discarded reusable capacity")
	}
}

func TestEncoderSuffixRejectsUnalignedPrefix(t *testing.T) {
	e := &encoder{chunkFrames: 100, windowChunks: 8, freq: [4]int{2}}
	const frames = 3300
	features := make([]float32, 2*frames)
	for _, skip := range []int{-1, 100, 800, 1601, frames} {
		if _, err := e.encodeSuffix(features, frames, skip, nil, nil); err == nil {
			t.Fatalf("accepted prefix %d without complete window/tile alignment", skip)
		}
	}
}

// Compare cached continuation with a full independent encoder at boundaries
// that change the final activation tile, the chunk padding, and attention
// windows. The per-frame values are identical as their row stride grows.
func TestEncoderPrefixMatchesFullExactly(t *testing.T) {
	m := loadModel(t, FormatF16)
	tr, err := NewTranscriber(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	e := m.enc
	ref, err := newEncoderWorkspace(e, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.close()
	step := e.prefixStep()
	for _, frames := range []int{step - 1, step, step + 1, step + 7, step + 8, step + 100, 2*step + 1, 2*step + 9, step + 5} {
		tr.features = ensure(tr.features, frames*e.freq[0])
		for band := range e.freq[0] {
			for f := range frames {
				tr.features[band*frames+f] = float32((band*13+f*7)%137)/64 - 1
			}
		}
		skip := tr.encPrefix.reusable(e, tr.features, frames)
		if frames == step+8 && skip != step {
			t.Fatalf("candidate path is not live: skip=%d", skip)
		}
		want := make([]float32, e.tokens(frames)*e.out)
		if _, err := e.encode(tr.features, frames, want, ref); err != nil {
			t.Fatal(err)
		}
		if err := tr.encodeContinuation(frames, true); err != nil {
			t.Fatal(err)
		}
		for i, v := range want {
			if math.Float32bits(v) != math.Float32bits(tr.embeds[i]) {
				t.Fatalf("frames=%d skip=%d output=%d: %08x != %08x", frames, skip, i, math.Float32bits(tr.embeds[i]), math.Float32bits(v))
			}
		}
	}
	frames := len(tr.features) / e.freq[0]
	if allocs := testing.AllocsPerRun(1, func() {
		if err := tr.encodeContinuation(frames, true); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warm encoder continuation allocations=%g", allocs)
	}
}

func TestEncoderPrefixGrowingPCMMatchesFullExactly(t *testing.T) {
	m := loadModel(t, FormatF16)
	tr, err := NewTranscriber(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	e := m.enc
	ref, err := newEncoderWorkspace(e, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.close()
	step := e.prefixStep()
	clip := clipPCM(t, "jfk")
	clip = clip[:len(clip)/160*160]
	pcm := make([]float32, (2*step+103)*160)
	for i := range pcm {
		pcm[i] = clip[i%len(clip)]
	}
	reused := 0
	for _, frames := range []int{step + 2, step + 3, step + 100, 2*step + 2, 2*step + 3, 2*step + 100} {
		tr.features = ensure(tr.features, frames*e.freq[0])
		if err := tr.frontend.Into(pcm[:frames*160], tr.features); err != nil {
			t.Fatal(err)
		}
		skip := tr.encPrefix.reusable(e, tr.features, frames)
		if skip > 0 {
			reused++
		}
		want := make([]float32, e.tokens(frames)*e.out)
		if _, err := e.encode(tr.features, frames, want, ref); err != nil {
			t.Fatal(err)
		}
		if err := tr.encodeContinuation(frames, true); err != nil {
			t.Fatal(err)
		}
		for i, v := range want {
			if math.Float32bits(v) != math.Float32bits(tr.embeds[i]) {
				t.Fatalf("frames=%d skip=%d output=%d: %08x != %08x", frames, skip, i, math.Float32bits(tr.embeds[i]), math.Float32bits(v))
			}
		}
		t.Logf("frames=%d reused_frames=%d", frames, skip)
	}
	if reused < 3 {
		t.Fatalf("reused only %d growing PCM prefixes", reused)
	}
	if tr.encPrefix.memory.Bytes() > encoderPrefixBytes {
		t.Fatal("feature snapshot exceeded its bound")
	}
	t.Logf("full encoder activation arena=%d bytes; incremental activation arena+snapshot=%d bytes", ref.memory.Bytes(), tr.enc.memory.Bytes()+tr.encPrefix.memory.Bytes())
	if allocs := testing.AllocsPerRun(1, func() {
		for _, frames := range []int{step + 2, step + 3, step + 100} {
			tr.features = ensure(tr.features, frames*e.freq[0])
			if err := tr.frontend.Into(pcm[:frames*160], tr.features); err != nil {
				panic(err)
			}
			if err := tr.encodeContinuation(frames, true); err != nil {
				panic(err)
			}
		}
	}); allocs != 0 {
		t.Fatalf("warm growing encoder allocations=%g", allocs)
	}
}
