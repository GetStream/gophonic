// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"github.com/GetStream/gophonic/internal/arena"
	"github.com/GetStream/gophonic/internal/whispergemm"
)

// newPackedPrefix reserves one bounded mapping. Pages are touched only for
// stored positions; an unused capacity is not eagerly zeroed by packing.
// The small matrix headers describe views and never own numeric payloads.
func (e *Evaluator) newPackedPrefix(capacity int) (*PrefixKV, error) {
	c := &e.m.cfg
	chunks := (capacity-1)/prefixChunk + 1
	keySize, err := whispergemm.PackedLen(c.headDim, capacity)
	if err != nil {
		return nil, err
	}
	valueSizes := make([]int, chunks)
	for ch := range chunks {
		valueSizes[ch], err = whispergemm.PackedLen(min(prefixChunk, capacity-ch*prefixChunk), c.headDim)
		if err != nil {
			return nil, err
		}
	}
	var lengths []int
	for range c.layers * c.kvHeads {
		lengths = append(lengths, keySize)
		lengths = append(lengths, valueSizes...)
	}
	memory, err := arena.New(lengths...)
	if err != nil {
		return nil, err
	}
	kv := &PrefixKV{owner: e, capacity: capacity, memory: memory,
		tokens: make([]int, 0, capacity), packs: make([]prefixPack, c.layers)}
	for l := range kv.packs {
		pk := &kv.packs[l]
		pk.keysT = make([]*whispergemm.PackedB, c.kvHeads)
		pk.values = make([][]*whispergemm.PackedB, c.kvHeads)
		for g := range c.kvHeads {
			pk.keysT[g], err = whispergemm.NewPackedBBuffer(c.headDim, capacity, memory.Take(keySize))
			if err != nil {
				_ = memory.Close()
				return nil, err
			}
			must(pk.keysT[g].Reshape(c.headDim, 0))
			pk.values[g] = make([]*whispergemm.PackedB, chunks)
			for ch, size := range valueSizes {
				rows := min(prefixChunk, capacity-ch*prefixChunk)
				pk.values[g][ch], err = whispergemm.NewPackedBRows(rows, c.headDim, memory.Take(size))
				if err != nil {
					_ = memory.Close()
					return nil, err
				}
			}
		}
	}
	return kv, nil
}

// copyPackedPrefix also implements truncation when src==kv. Whole value
// chunks stay untouched in that case; only a partial tail changes its logical row count.
func (kv *PrefixKV) copyPackedPrefix(src *PrefixKV, positions int) {
	for l := range kv.packs {
		dst, from := &kv.packs[l], &src.packs[l]
		for g := range dst.keysT {
			must(dst.keysT[g].CopyColumnsFrom(from.keysT[g], positions))
			for ch := 0; ch*prefixChunk < positions; ch++ {
				must(dst.values[g][ch].CopyRowsFrom(from.values[g][ch], min(prefixChunk, positions-ch*prefixChunk)))
			}
		}
	}
}
