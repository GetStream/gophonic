// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !goexperiment.simd || !arm64

package qwen3lm

func swigluInto(gate, up []float32) {
	up = up[:len(gate)]
	for i, x := range gate {
		gate[i] = silu32(x) * up[i]
	}
}

func sumSquares(x []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < len(x); i += 4 {
		s0 += x[i] * x[i]
		s1 += x[i+1] * x[i+1]
		s2 += x[i+2] * x[i+2]
		s3 += x[i+3] * x[i+3]
	}
	for ; i < len(x); i++ {
		s0 += x[i] * x[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func scaleMulInto(dst, src, weight []float32, inv float32) {
	src, weight = src[:len(dst)], weight[:len(dst)]
	for i := range dst {
		dst[i] = src[i] * inv * weight[i]
	}
}

func addInto(dst, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] += src[i]
	}
}

func dot32(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func axpy32(dst, x []float32, a float32) {
	x = x[:len(dst)]
	for i := range dst {
		dst[i] += a * x[i]
	}
}

func scaleVector(v []float32, a float32) {
	for i := range v {
		v[i] *= a
	}
}

func rotateHalves(x1, x2, cos, sin []float32) {
	x2, cos, sin = x2[:len(x1)], cos[:len(x1)], sin[:len(x1)]
	for i := range x1 {
		a, b := x1[i], x2[i]
		x1[i] = a*cos[i] - b*sin[i]
		x2[i] = b*cos[i] + a*sin[i]
	}
}

// softmaxScaled replaces row with softmax(row*scale).
func softmaxScaled(row []float32, scale float32) {
	m := row[0]
	for _, v := range row {
		m = max(m, v)
	}
	m *= scale
	var sum float32
	for i, v := range row {
		row[i] = expNonPositive32(v*scale - m)
		sum += row[i]
	}
	scaleVector(row, 1/sum)
}

// butterflies replaces (a, b) with (a+b, a-b) elementwise.
func butterflies(a, b []float32) {
	b = b[:len(a)]
	for i := range a {
		a[i], b[i] = a[i]+b[i], a[i]-b[i]
	}
}
