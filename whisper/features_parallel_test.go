// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"testing"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

// Sharding the frontend must not change a single output bit.
func TestFeaturesParallelMatchesSerial(t *testing.T) {
	pcm := make([]float32, 16000*17+123)
	state := uint32(9)
	for i := range pcm {
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5
		pcm[i] = float32(int32(state))/float32(math.MaxInt32)*0.3 + float32(math.Sin(float64(i)*0.01))*0.2
	}
	want := make([]float32, MelBins*MelFrames)
	if err := FeaturesInto(pcm, want, NewFeatureWorkspace()); err != nil {
		t.Fatal(err)
	}
	for _, workers := range []int{2, 3, 8, 13} {
		e, err := whispergemm.NewExecutor(workers)
		if err != nil {
			t.Fatal(err)
		}
		w := NewFeatureWorkspace()
		got := make([]float32, len(want))
		for run := 0; run < 2; run++ {
			if err := featuresInto(pcm, got, w, e); err != nil {
				t.Fatal(err)
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("workers=%d index %d: %v != %v", workers, i, got[i], want[i])
				}
			}
		}
		if allocs := testing.AllocsPerRun(3, func() { _ = featuresInto(pcm, got, w, e) }); allocs != 0 {
			t.Fatalf("workers=%d allocations %g", workers, allocs)
		}
		e.Close()
	}
}
