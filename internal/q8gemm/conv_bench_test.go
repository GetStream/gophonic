// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause


package q8gemm

import "testing"

func BenchmarkPackRange4096x12(b *testing.B) {
	ws, _ := NewWorkspace(4096)
	x := make([]float32, 12*4096)
	for i := range x {
		x[i] = float32(i%97) - 48
	}
	for b.Loop() {
		_ = ws.Prepare(12, 4096)
		_ = ws.PackRange(x, 4096, 0, 4096)
	}
}
