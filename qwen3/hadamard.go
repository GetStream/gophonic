// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import "math"

// maxRotationBlock bounds the Hadamard block. Qwen3-8B's 4096- and
// 12288-wide projection inputs use 4096-value blocks (12288 as three).
const maxRotationBlock = 4096

// rotation is a randomized Hadamard transform R = H·diag(signs), applied
// blockwise and normalized, so R is orthogonal. Projections in the int8
// format store W·Rᵀ and multiply by R·x: the product W·x is unchanged before
// rounding, while R spreads activation outliers across every channel so that
// one int8 scale per row loses little precision.
type rotation struct {
	signs []float32
	block int
	scale float32
}

// newRotation returns the rotation for k-wide inputs. The signs come from a
// fixed seed per width, so loading the same checkpoint always yields the same
// weights.
func newRotation(k int) *rotation {
	block := 1
	for block < maxRotationBlock && k%(2*block) == 0 {
		block *= 2
	}
	r := &rotation{signs: make([]float32, k), block: block, scale: float32(1 / math.Sqrt(float64(block)))}
	state := uint64(0x9e3779b97f4a7c15) ^ uint64(k)
	for i := range r.signs {
		// splitmix64
		state += 0x9e3779b97f4a7c15
		z := state
		z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
		z = (z ^ z>>27) * 0x94d049bb133111eb
		z ^= z >> 31
		r.signs[i] = 1
		if z&1 == 1 {
			r.signs[i] = -1
		}
	}
	return r
}

// apply rotates x in place; len(x) must equal the rotation's width.
func (r *rotation) apply(x []float32) {
	x = x[:len(r.signs)]
	for b := 0; b < len(x); b += r.block {
		v := x[b : b+r.block]
		scaleMulInto(v, v, r.signs[b:b+r.block], r.scale)
		fwht(v)
	}
}

// fwht is the unnormalized in-place Walsh–Hadamard transform of a
// power-of-two-length vector.
func fwht(v []float32) {
	n := len(v)
	h := 1
	for ; h < 4 && h < n; h *= 2 {
		for i := 0; i < n; i += 2 * h {
			for j := i; j < i+h; j++ {
				a, b := v[j], v[j+h]
				v[j], v[j+h] = a+b, a-b
			}
		}
	}
	for ; h < n; h *= 2 {
		for i := 0; i < n; i += 2 * h {
			butterflies(v[i:i+h], v[i+h:i+2*h])
		}
	}
}

// unapply applies the inverse rotation Rᵀ in place.
func (r *rotation) unapply(x []float32) {
	x = x[:len(r.signs)]
	for b := 0; b < len(x); b += r.block {
		fwht(x[b : b+r.block])
	}
	scaleMulInto(x, x, r.signs, r.scale)
}

// applyRows replaces the n×k row-major matrix a with R·a, rotating along
// the row index; n must not exceed the rotation's width.
func (r *rotation) applyRows(a []float32, n, k int) {
	for i := range n {
		scaleVector(a[i*k:(i+1)*k], r.signs[i])
	}
	for b := 0; b < n; b += r.block {
		for h := 1; h < r.block; h *= 2 {
			for i := b; i < b+r.block; i += 2 * h {
				for j := i; j < i+h; j++ {
					butterflies(a[j*k:(j+1)*k], a[(j+h)*k:(j+h+1)*k])
				}
			}
		}
	}
	scaleVector(a[:n*k], r.scale)
}
