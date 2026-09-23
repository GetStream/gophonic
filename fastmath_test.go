// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"math"
	"testing"
)

func TestExpNegative32Accuracy(t *testing.T) {
	for i := 0; i <= 80000; i++ {
		x := -float32(i) / 1000
		want := float32(math.Exp(float64(x)))
		got := expNegative32(x)
		if delta := math.Abs(float64(got - want)); delta > 2e-6 {
			t.Fatalf("exp(%.6g): got %.9g, want %.9g, delta %.3g", x, got, want, delta)
		}
	}
}

func TestErfApprox32Accuracy(t *testing.T) {
	for i := -8000; i <= 8000; i++ {
		x := float32(i) / 1000
		want := float32(math.Erf(float64(x)))
		got := erfApprox32(x)
		if delta := math.Abs(float64(got - want)); delta > 2e-6 {
			t.Fatalf("erf(%.6g): got %.9g, want %.9g, delta %.3g", x, got, want, delta)
		}
	}
}

func TestTanhApprox32Accuracy(t *testing.T) {
	for i := -10000; i <= 10000; i++ {
		x := float32(i) / 1000
		want := float32(math.Tanh(float64(x)))
		got := tanhApprox32(x)
		if delta := math.Abs(float64(got - want)); delta > 2e-6 {
			t.Fatalf("tanh(%.6g): got %.9g, want %.9g, delta %.3g", x, got, want, delta)
		}
	}
}

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
