// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

// applyBiasGELU adds an optional per-column bias to each row, then applies
// the exact-erf GELU, in one pass where the NEON kernel is available.
func applyBiasGELU(values, bias []float32, rows, width int) {
	if geluAccelerated && width > 0 && width%4 == 0 && (bias == nil || len(bias) >= width) {
		var b *float32
		if bias != nil {
			b = &bias[0]
		}
		for r := 0; r < rows; r++ {
			geluNEON(&values[r*width], b, width, &geluNormalTable)
		}
		return
	}
	addRowBias(values, bias, rows, width)
	applyGELU(values[:rows*width])
}
