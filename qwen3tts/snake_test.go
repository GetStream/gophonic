// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestSnakeMatchesMathSin(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	n := 96
	s := snake{a: make([]float32, n), invB: make([]float32, n)}
	for i := range n {
		s.a[i], s.invB[i] = float32(math.Exp(r.NormFloat64())), float32(r.Float64()*3)
	}
	src := make([]float32, 64*n)
	for i := range src {
		src[i] = float32(r.NormFloat64() * 4)
	}
	dst := make([]float32, len(src))
	s.apply(dst, src, n)
	var worst float64
	for i, v := range src {
		c := i % n
		sv := math.Sin(float64(v) * float64(s.a[c]))
		want := float64(v) + float64(s.invB[c])*sv*sv
		worst = max(worst, math.Abs(want-float64(dst[i]))/(1+math.Abs(want)))
	}
	if worst > 1e-6 {
		t.Fatalf("SnakeBeta relative error %g", worst)
	}
	if allocs := testing.AllocsPerRun(10, func() { s.apply(dst, src, n) }); allocs != 0 {
		t.Fatalf("SnakeBeta allocates %v times", allocs)
	}
}
