// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

// Weights loaded without a head load it when first needed, as they would
// have with it.
func TestLoadHeadLater(t *testing.T) {
	for _, shape := range []lmtest.Geometry{lmtest.Shape, lmtest.GPUShape} {
		ck := lmtest.WriteShape(t, 23, shape, nil)
		formats := []string{WeightsF16, WeightsInt8}
		if shape == lmtest.GPUShape {
			if !GPUAvailable() {
				continue
			}
			formats = []string{WeightsGPU, WeightsGPUQ8}
		}
		ids := []int{4, 8, 15, 16, 22, 3}
		want := ck.ReferenceLogits(ck.ReferenceHidden(ids))
		for _, format := range formats {
			m, err := Load(ck.Dir, LoadOptions{Format: format})
			if err != nil {
				t.Fatal(err)
			}
			e, _ := NewEvaluator(m)
			ws, _ := e.NewWorkspace(2)
			hidden, logits := make([]float32, shape.Hidden), make([]float32, shape.Vocab)
			if err := e.HiddenLastInto(ids, hidden, ws); err != nil {
				t.Fatal(err)
			}
			if m.HasHead() || e.LogitsInto(hidden, logits, ws) == nil {
				t.Fatalf("%s: logits without a head", format)
			}
			for range 2 { // the second load finds it loaded
				if err := m.LoadHead("lm_head.weight"); err != nil {
					t.Fatal(err)
				}
			}
			if err := m.LoadHead("model.embed_tokens.weight"); err == nil {
				t.Fatalf("%s: a second, different head loaded", format)
			}
			if err := e.LogitsInto(hidden, logits, ws); err != nil {
				t.Fatal(err)
			}
			if cos, _ := lmtest.VectorParity(logits, want); !m.HasHead() || cos < 0.999 {
				t.Fatalf("%s: logits cosine %.6f to the reference", format, cos)
			}
			ws.Close()
			m.Release()
		}
	}
}
