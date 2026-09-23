// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor

import "math"

// expNegative32 evaluates exp(x) for the nonpositive arguments used by GELU
// and softmax. Range reduction keeps the polynomial on [-ln(2), 0].
func expNegative32(x float32) float32 {
	if x <= -80 {
		return 0
	}
	n := int(x * 1.4426950408889634)
	r := x - float32(n)*0.6931471805599453
	p := float32(1.0 / 40320.0)
	p = 1.0/5040.0 + r*p
	p = 1.0/720.0 + r*p
	p = 1.0/120.0 + r*p
	p = 1.0/24.0 + r*p
	p = 1.0/6.0 + r*p
	p = 0.5 + r*p
	p = 1 + r*p
	p = 1 + r*p
	return math.Float32frombits(uint32(n+127)<<23) * p
}

// erfApprox32 uses the standard five-term approximation with a maximum
// absolute error near 1.5e-7 when its exponential is exact.
func erfApprox32(x float32) float32 {
	sign := float32(1)
	if x < 0 {
		sign = -1
		x = -x
	}
	t := 1 / (1 + 0.3275911*x)
	p := float32(1.061405429)
	p = -1.453152027 + t*p
	p = 1.421413741 + t*p
	p = -0.284496736 + t*p
	p = 0.254829592 + t*p
	return sign * (1 - t*p*expNegative32(-x*x))
}

func tanhApprox32(x float32) float32 {
	if x < 0 {
		e := expNegative32(2 * x)
		return (e - 1) / (e + 1)
	}
	e := expNegative32(-2 * x)
	return (1 - e) / (1 + e)
}
