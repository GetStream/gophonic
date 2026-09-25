// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

// SnakeBeta needs sin² for millions of values per second of speech. sin²
// has period π, so the argument is reduced to [-π/2, π/2] with π split in
// three (Cody–Waite), and sin is the odd Taylor polynomial of degree 11,
// accurate to 6e-8 there.
const (
	invPi = 0.31830988618379067
	pi1   = 3.140625
	pi2   = 9.67502593994140625e-4
	pi3   = 1.509957990978376432e-7
	s3    = -1.0 / 6
	s5    = 1.0 / 120
	s7    = -1.0 / 5040
	s9    = 1.0 / 362880
	s11   = -1.0 / 39916800
)

// sin2 returns sin²(y).
func sin2(y float32) float32 {
	k := roundEven(y * invPi)
	r := y - k*pi1 - k*pi2 - k*pi3
	r2 := r * r
	s := r + r*r2*(s3+r2*(s5+r2*(s7+r2*(s9+r2*s11))))
	return s * s
}

func roundEven(x float32) float32 {
	const shift = 1 << 23
	if x >= 0 {
		return (x + shift) - shift
	}
	return (x - shift) + shift
}

// snakeScalar writes SnakeBeta of rows of n channels from src, plus bias
// when given, to dst.
func snakeScalar(dst, src, bias, a, invB []float32, n int) {
	for r := 0; r < len(src); r += n {
		x, y := src[r:r+n], dst[r:r+n]
		for i, v := range x {
			if bias != nil {
				v += bias[i]
			}
			y[i] = v + invB[i]*sin2(v*a[i])
		}
	}
}

func addToScalar(dst, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] += src[i]
	}
}

// residualScalar writes rows of n channels of src + z + bias to dst.
func residualScalar(dst, src, z, bias []float32, n int) {
	for r := 0; r < len(dst); r += n {
		x, y, o := src[r:r+n], z[r:r+n], dst[r:r+n]
		for i := range o {
			o[i] = x[i] + y[i] + bias[i]
		}
	}
}
