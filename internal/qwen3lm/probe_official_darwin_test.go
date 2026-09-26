// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"math"
	"testing"
)

// The GPU's probe weighs keys as the CPU's does, on a real checkpoint and
// every GPU format.
func TestOfficialGPUProbeMatchesCPU(t *testing.T) {
	heads := [][2]int{{0, 0}, {3, 0}, {3, 7}, {17, 31}, {35, 12}}
	probes := func(format string) (ids []int, out [][]float32) {
		lm := loadOfficialLM(t, format)
		ids, err := lm.tokens.EncodeInto("The quick brown fox jumps over the lazy dog while the band plays a slow song.", make([]int, 0, 64), &lm.tokenWS)
		if err != nil {
			t.Fatal(err)
		}
		kv, err := lm.eval.NewPrefixKV(128)
		if err != nil {
			t.Fatal(err)
		}
		defer kv.Close()
		hidden := make([]float32, lm.hidden)
		n := len(ids)
		for _, lh := range heads {
			if err := lm.eval.HiddenLastExtendInto(kv, 0, ids[:n-1], hidden, lm.ws); err != nil {
				t.Fatal(err)
			}
			p := Probe{Layer: lh[0], Head: lh[1], Probs: make([]float32, n)}
			if err := lm.eval.HiddenLastExtendProbeInto(kv, n-1, ids[n-1:], Embeds{}, hidden, &p, lm.ws); err != nil {
				t.Fatal(err)
			}
			out = append(out, p.Probs)
		}
		lm.weights.Release()
		return ids, out
	}
	_, cpu := probes(WeightsF16)
	for _, format := range []string{WeightsGPU, WeightsGPUQ8, WeightsGPUQ4} {
		_, gpu := probes(format)
		for h, lh := range heads {
			var worst, sum float64
			for j := range cpu[h] {
				worst = max(worst, math.Abs(float64(gpu[h][j]-cpu[h][j])))
				sum += float64(gpu[h][j])
			}
			t.Logf("%s layer %d head %d: max difference %.4f, sum %.6f", format, lh[0], lh[1], worst, sum)
			if tol := map[string]float64{WeightsGPUQ4: 0.1}[format]; worst > max(tol, 0.03) || math.Abs(sum-1) > 1e-4 {
				t.Fatalf("%s layer %d head %d: GPU probabilities %v, CPU %v", format, lh[0], lh[1], gpu[h], cpu[h])
			}
		}
	}
}
