// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whisper

import "simd/archsimd"

// Go's amd64 archsimd API does not expose the ARM64 reduction methods.
func maxFloat4(v archsimd.Float32x4) float32 {
	return max(max(v.GetElem(0), v.GetElem(1)), max(v.GetElem(2), v.GetElem(3)))
}

func anyMask4(v archsimd.Int32x4) bool {
	return v.GetElem(0)|v.GetElem(1)|v.GetElem(2)|v.GetElem(3) != 0
}

func allMask4(v archsimd.Int32x4) bool {
	return v.GetElem(0)&v.GetElem(1)&v.GetElem(2)&v.GetElem(3) == -1
}
