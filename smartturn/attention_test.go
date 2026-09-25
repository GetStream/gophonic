// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package smartturn

import (
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/vec"
)

func TestAttentionTilesMatchScalarReference(t *testing.T) {
	q := make([]float32, sequenceLength*hiddenSize)
	k := make([]float32, sequenceLength*hiddenSize)
	v := make([]float32, sequenceLength*hiddenSize)
	for i := range q {
		q[i] = float32(math.Sin(float64(i*7+1))) * 2.5
		k[i] = float32(math.Cos(float64(i*3+2))) * 2.5
		v[i] = float32(math.Sin(float64(i*11+3))) * 0.2
	}
	scores := make([]float32, attentionHeads*sequenceLength*sequenceLength)
	out := make([]float32, sequenceLength*hiddenSize)
	job := parallelJob{q: q, k: k, v: v, scores: scores, output: out}
	var firstTile [16]float32
	vec.Dot4x4(
		q[:headSize], q[hiddenSize:hiddenSize+headSize], q[2*hiddenSize:2*hiddenSize+headSize], q[3*hiddenSize:3*hiddenSize+headSize],
		k[:headSize], k[hiddenSize:hiddenSize+headSize], k[2*hiddenSize:2*hiddenSize+headSize], k[3*hiddenSize:3*hiddenSize+headSize], &firstTile,
	)
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			var want float32
			for d := 0; d < headSize; d++ {
				want += q[r*hiddenSize+d] * k[c*hiddenSize+d]
			}
			if delta := math.Abs(float64(firstTile[r*4+c] - want)); delta > 2e-4 {
				t.Fatalf("first dot tile[%d,%d] got %.9g want %.9g delta %.3g", r, c, firstTile[r*4+c], want, delta)
			}
		}
	}
	runAttentionRange(&job, 0, attentionHeads*(sequenceLength/4))
	for head := 0; head < attentionHeads; head++ {
		for query := 0; query < sequenceLength; query++ {
			row := (query*attentionHeads + head) * sequenceLength
			wantScores := make([]float32, sequenceLength)
			for key := 0; key < sequenceLength; key++ {
				qv := q[query*hiddenSize+head*headSize : query*hiddenSize+(head+1)*headSize]
				kv := k[key*hiddenSize+head*headSize : key*hiddenSize+(head+1)*headSize]
				wantScores[key] = vec.Dot(qv, kv) / 8
			}
			softmaxInPlace(wantScores)
			for key := range wantScores {
				if delta := math.Abs(float64(scores[row+key] - wantScores[key])); delta > 2e-6 {
					t.Fatalf("attention score h=%d q=%d k=%d got %.9g want %.9g delta %.3g", head, query, key, scores[row+key], wantScores[key], delta)
				}
			}
			var wantContext [headSize]float32
			for key, weight := range wantScores {
				for d := 0; d < headSize; d++ {
					wantContext[d] += weight * v[key*hiddenSize+head*headSize+d]
				}
			}
			gotContext := out[query*hiddenSize+head*headSize : query*hiddenSize+(head+1)*headSize]
			for d := range wantContext {
				if delta := math.Abs(float64(gotContext[d] - wantContext[d])); delta > 2e-6 {
					t.Fatalf("attention context h=%d q=%d d=%d got %.9g want %.9g delta %.3g", head, query, d, gotContext[d], wantContext[d], delta)
				}
			}
		}
	}
}
