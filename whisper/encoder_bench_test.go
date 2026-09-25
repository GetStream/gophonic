// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
)

var encoderBenchmarkSink float32

// BenchmarkEncoder measures complete warm audio encoding, including stem,
// attention, softmax, GELU and final normalization. Model loading, workspace
// construction and three warm-up passes (weight packing and worker startup)
// are excluded explicitly.
func BenchmarkEncoder(b *testing.B) {
	path := testmodels.Path(b, testmodels.WhisperTinyEN)
	m, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	mel, err := readFloatFixture(filepath.Join("..", "testdata", "whisper", "jfk.mel.f32le"), MelBins*MelFrames)
	if err != nil {
		b.Fatal(err)
	}
	for _, workers := range []int{1, 2, 4, 8, 12, 16} {
		b.Run(fmt.Sprintf("workers%d", workers), func(b *testing.B) {
			w, err := NewEncoderWorkspaceWithWorkers(workers)
			if err != nil {
				b.Fatal(err)
			}
			defer w.Close()
			dst := make([]float32, AudioFrames*AudioState)
			for range 3 {
				if err := m.EncodeInto(mel, dst, w); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			for b.Loop() {
				if err := m.EncodeInto(mel, dst, w); err != nil {
					b.Fatal(err)
				}
			}
			encoderBenchmarkSink = dst[len(dst)-1]
		})
	}
}
