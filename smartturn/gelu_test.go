// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package smartturn

import (
	"math"
	"testing"
)

func BenchmarkGelu(b *testing.B) {
	var sum float32
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sum += gelu(float32(i&8191)/512 - 8)
	}
	if sum == 123 {
		b.Fatal(sum)
	}
}

func BenchmarkGeluStd(b *testing.B) {
	var sum float32
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		x := float32(i&8191)/512 - 8
		sum += 0.5 * x * (1 + float32(math.Erf(float64(x)*0.7071067811865475244)))
	}
	if sum == 123 {
		b.Fatal(sum)
	}
}
