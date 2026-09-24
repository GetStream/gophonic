// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"hash/maphash"
	"unsafe"
)

// embeddingCache maps token sequences to finished hidden vectors. Qwen3
// inference is deterministic for a given model and token sequence, so a hit
// returns exactly what a recomputation would. Keys are two independently
// seeded 64-bit hashes of the token IDs plus their count (a false match needs
// a 128-bit collision). Storage is fixed at construction: an open-addressing
// index with linear probing and CLOCK eviction, so lookups and inserts never
// allocate.
type embeddingCache struct {
	width     int
	seeds     [2]maphash.Seed
	keys      []cacheKey
	vectors   []float32 // [entries][width]
	ref       []bool    // CLOCK reference bits
	slots     []int32   // index table: entry+1, or 0 when empty
	hand      int
	used      int
	hits, all uint64
}

type cacheKey struct {
	h0, h1 uint64
	n      int
}

func newEmbeddingCache(entries, width int) *embeddingCache {
	if entries <= 0 {
		return nil
	}
	slots := 1
	for slots < 2*entries {
		slots <<= 1
	}
	return &embeddingCache{
		width:   width,
		seeds:   [2]maphash.Seed{maphash.MakeSeed(), maphash.MakeSeed()},
		keys:    make([]cacheKey, entries),
		vectors: make([]float32, entries*width),
		ref:     make([]bool, entries),
		slots:   make([]int32, slots),
	}
}

func (c *embeddingCache) key(ids []int) cacheKey {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(ids))), len(ids)*int(unsafe.Sizeof(ids[0])))
	return cacheKey{maphash.Bytes(c.seeds[0], b), maphash.Bytes(c.seeds[1], b), len(ids)}
}

// find returns the slot holding k, or the empty slot where it would go.
func (c *embeddingCache) find(k cacheKey) (slot int, entry int) {
	mask := len(c.slots) - 1
	for s := int(k.h0) & mask; ; s = (s + 1) & mask {
		e := int(c.slots[s]) - 1
		if e < 0 || c.keys[e] == k {
			return s, e
		}
	}
}

// get copies the cached vector for ids into dst and reports whether it hit.
func (c *embeddingCache) get(ids []int, dst []float32) bool {
	c.all++
	_, e := c.find(c.key(ids))
	if e < 0 {
		return false
	}
	c.hits++
	c.ref[e] = true
	copy(dst, c.vectors[e*c.width:(e+1)*c.width])
	return true
}

// put stores a copy of vec for ids, evicting with CLOCK when full.
func (c *embeddingCache) put(ids []int, vec []float32) {
	k := c.key(ids)
	slot, e := c.find(k)
	if e >= 0 {
		return
	}
	if c.used < len(c.keys) {
		e = c.used
		c.used++
	} else {
		for c.ref[c.hand] {
			c.ref[c.hand] = false
			c.hand = (c.hand + 1) % len(c.keys)
		}
		e = c.hand
		c.hand = (c.hand + 1) % len(c.keys)
		c.remove(c.keys[e])
		slot, _ = c.find(k) // removal may have shifted the probe chain
	}
	c.keys[e], c.ref[e] = k, false
	c.slots[slot] = int32(e + 1)
	copy(c.vectors[e*c.width:(e+1)*c.width], vec)
}

// remove deletes k's index slot with backward-shift deletion, which keeps
// every remaining probe chain contiguous without tombstones.
func (c *embeddingCache) remove(k cacheKey) {
	mask := len(c.slots) - 1
	s, e := c.find(k)
	if e < 0 {
		return
	}
	for {
		c.slots[s] = 0
		next := s
		for {
			next = (next + 1) & mask
			ne := int(c.slots[next]) - 1
			if ne < 0 {
				return
			}
			home := int(c.keys[ne].h0) & mask
			// Move the entry back if its home is not in (s, next].
			if (next > s && (home <= s || home > next)) || (next < s && home <= s && home > next) {
				c.slots[s] = c.slots[next]
				s = next
				break
			}
		}
	}
}
