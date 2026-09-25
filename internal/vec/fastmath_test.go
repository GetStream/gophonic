// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package vec

import (
	"math"
	"testing"
)

func TestExpNegative32Accuracy(t *testing.T) {
	for i := 0; i <= 80000; i++ {
		x := -float32(i) / 1000
		want := float32(math.Exp(float64(x)))
		got := ExpNegative(x)
		if delta := math.Abs(float64(got - want)); delta > 2e-6 {
			t.Fatalf("exp(%.6g): got %.9g, want %.9g, delta %.3g", x, got, want, delta)
		}
	}
}

func TestErfApprox32Accuracy(t *testing.T) {
	for i := -8000; i <= 8000; i++ {
		x := float32(i) / 1000
		want := float32(math.Erf(float64(x)))
		got := Erf(x)
		if delta := math.Abs(float64(got - want)); delta > 2e-6 {
			t.Fatalf("erf(%.6g): got %.9g, want %.9g, delta %.3g", x, got, want, delta)
		}
	}
}

func TestTanhApprox32Accuracy(t *testing.T) {
	for i := -10000; i <= 10000; i++ {
		x := float32(i) / 1000
		want := float32(math.Tanh(float64(x)))
		got := Tanh(x)
		if delta := math.Abs(float64(got - want)); delta > 2e-6 {
			t.Fatalf("tanh(%.6g): got %.9g, want %.9g, delta %.3g", x, got, want, delta)
		}
	}
}
