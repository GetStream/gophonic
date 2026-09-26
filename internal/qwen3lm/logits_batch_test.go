// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
)

func TestLogitsRowsMatchesSeparateCallsExactly(t *testing.T) {
	ck := lmtest.Write(t, 47)
	for _, format := range []string{WeightsF16, WeightsInt8} {
		m, err := Load(ck.Dir, LoadOptions{Format: format, Head: "lm_head.weight"})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(m.Release)
		e, _ := NewEvaluator(m)
		for _, workers := range []int{1, 3, 8} {
			t.Run(fmt.Sprintf("%s/workers%d", format, workers), func(t *testing.T) {
				ws, err := e.NewWorkspace(workers)
				if err != nil {
					t.Fatal(err)
				}
				defer ws.Close()
				reference, err := e.NewWorkspace(1)
				if err != nil {
					t.Fatal(err)
				}
				defer reference.Close()
				rng := rand.New(rand.NewSource(53))
				for _, rows := range []int{1, 2, 3, 8, 16, 17, 31, 32, 33, maxTail, 1} {
					h, n := m.cfg.hidden, m.cfg.vocab
					hidden := make([]float32, rows*h)
					for i := range hidden {
						hidden[i] = float32(rng.NormFloat64()) * float32(math.Ldexp(1, (i/h%3-1)*20))
					}
					want, got := make([]float32, rows*n), make([]float32, rows*n)
					for r := range rows {
						if err := e.LogitsInto(hidden[r*h:(r+1)*h], want[r*n:(r+1)*n], reference); err != nil {
							t.Fatal(err)
						}
					}
					if err := e.LogitsRowsInto(hidden, got, ws); err != nil {
						t.Fatal(err)
					}
					for i, v := range got {
						if math.Float32bits(v) != math.Float32bits(want[i]) {
							t.Fatalf("rows%d index%d: %08x != %08x", rows, i, math.Float32bits(v), math.Float32bits(want[i]))
						}
					}
					if ws.op.exactRows {
						t.Fatal("row batch mode leaked into next operation")
					}
					if a := testing.AllocsPerRun(1, func() {
						if err := e.LogitsRowsInto(hidden, got, ws); err != nil {
							panic(err)
						}
					}); a != 0 {
						t.Fatalf("rows%d allocs=%g", rows, a)
					}
				}
			})
		}
	}
}
