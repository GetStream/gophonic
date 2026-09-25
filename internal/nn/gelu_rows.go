// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package nn

// BiasGELU adds an optional per-column bias to each row, then applies
// the exact-erf GELU, in one pass where the NEON kernel is available.
func BiasGELU(values, bias []float32, rows, width int) {
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
	AddRowBias(values, bias, rows, width)
	GELU(values[:rows*width])
}

// AddRowBias adds bias to each of rows rows of width values; a nil bias
// adds nothing.
func AddRowBias(values, bias []float32, rows, width int) {
	if bias == nil {
		return
	}
	for row := 0; row < rows; row++ {
		start := row * width
		for col, value := range bias {
			values[start+col] += value
		}
	}
}
