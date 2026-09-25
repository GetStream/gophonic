// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"math"
	"math/rand/v2"
	"unsafe"
)

// sampler draws from the top k logits after a temperature; its scratch is
// reused, so draws allocate nothing.
type sampler struct {
	pcg  rand.PCG
	rng  *rand.Rand
	idx  []int32
	prob []float32
}

func (s *sampler) seed(v uint64) {
	s.pcg.Seed(v, v^0x9e3779b97f4a7c15)
	if s.rng == nil {
		s.rng = rand.New(&s.pcg)
	}
}

func (s *sampler) topK(logits []float32, k int, temp float32) int {
	k = min(k, len(logits))
	if cap(s.idx) < k {
		s.idx, s.prob = make([]int32, k), make([]float32, k)
	}
	s.idx, s.prob = s.idx[:k], s.prob[:k]
	// A min-heap of the best k.
	for i := range k {
		s.idx[i], s.prob[i] = int32(i), logits[i]
		for j := i; j > 0; {
			p := (j - 1) / 2
			if s.prob[p] <= s.prob[j] {
				break
			}
			s.swap(j, p)
			j = p
		}
	}
	for i := k; i < len(logits); i++ {
		if logits[i] > s.prob[0] {
			s.idx[0], s.prob[0] = int32(i), logits[i]
			s.down(0, k)
		}
	}
	top := s.prob[0]
	for i := 1; i < k; i++ {
		top = max(top, s.prob[i])
	}
	var sum float32
	for i := range k {
		p := float32(math.Exp(float64((s.prob[i] - top) / temp)))
		s.prob[i] = p
		sum += p
	}
	r := s.rng.Float32() * sum
	for i := range k {
		r -= s.prob[i]
		if r <= 0 {
			return int(s.idx[i])
		}
	}
	return int(s.idx[k-1])
}

func (s *sampler) down(j, n int) {
	for {
		l := 2*j + 1
		if l >= n {
			return
		}
		m := l
		if r := l + 1; r < n && s.prob[r] < s.prob[l] {
			m = r
		}
		if s.prob[j] <= s.prob[m] {
			return
		}
		s.swap(j, m)
		j = m
	}
}

func (s *sampler) swap(a, b int) {
	s.idx[a], s.idx[b] = s.idx[b], s.idx[a]
	s.prob[a], s.prob[b] = s.prob[b], s.prob[a]
}

// stringOf views b as a string without copying; b must not change while
// the string is used.
func stringOf(b []byte) string { return unsafe.String(unsafe.SliceData(b), len(b)) }
