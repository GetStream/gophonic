// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build goexperiment.simd && amd64 && amd64.v3

package whispergemm

import "testing"

// Compare the production four-row tile with two two-row tiles in the same
// process, avoiding cross-run CPU and clock differences on hosted runners.
func BenchmarkAMD64Tile(b *testing.B) {
	const m, k, n = 1500, 384, 384
	for _, variant := range []struct {
		name string
		mul  func([]float32, int, []float32, int, []float32, int, int, int)
	}{
		{"4x16", mulPacked},
		{"2x16", mulPacked2x16Benchmark},
	} {
		b.Run(variant.name, func(b *testing.B) {
			a, _, dst, packed := benchmarkData(b, m, k, n)
			b.ReportAllocs()
			for b.Loop() {
				variant.mul(dst, n, a, k, packed.data, m, k, n)
			}
			benchmarkSink = dst[len(dst)-1]
		})
	}
}

func mulPacked2x16Benchmark(dst []float32, dstStride int, a []float32, aStride int, packed []float32, m, k, n int) {
	for c := 0; c < n; c += panelColumns {
		weights := packed[c*k : (c+panelColumns)*k]
		for r := 0; r < m; r += 2 {
			kernel2x16AMD64(a[r*aStride:r*aStride+k], a[(r+1)*aStride:(r+1)*aStride+k], weights,
				dst[r*dstStride+c:r*dstStride+c+panelColumns], dst[(r+1)*dstStride+c:(r+1)*dstStride+c+panelColumns])
		}
	}
}
