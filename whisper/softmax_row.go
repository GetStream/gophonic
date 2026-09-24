// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

// softmaxExpRow replaces x with exp(x-max(x)) and returns 1/sum. Callers
// fold the normalization into a later, narrower product. Inputs are clamped
// at ln(2^-126) after subtracting the maximum; clamped terms are below
// 1.2e-38 against a sum of at least one.
func softmaxExpRow(x []float32) float32 {
	if softmaxAccelerated && len(x) >= 4 && len(x)%4 == 0 {
		return 1 / softmaxExpNEON(&x[0], len(x), &softmaxConstants)
	}
	return softmaxExpRowFallback(x)
}
