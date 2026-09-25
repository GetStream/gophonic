// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clm

import (
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

func BenchmarkOfficialHeadScore(b *testing.B) {
	path := testmodels.Path(b, testmodels.CLMHead)
	h, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	ws := h.NewWorkspace()
	state := make([]float32, h.Config().EncoderDim)
	action := make([]float32, len(state))
	for i := range state {
		state[i] = float32(i%41-20) / 17
		action[i] = float32(i%37-18) / 13
	}
	actions := [][]float32{action}
	scores := make([]float32, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := h.ScoreInto(state, actions, 1, scores, ws); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	benchmarkScoreSink = scores[0]
}

var benchmarkScoreSink float32
