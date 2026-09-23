// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/GetStream/gophonic/internal/whispergemm"
)

var decoderBenchmarkSink int

func decoderBenchmarkFixture(b *testing.B) (*Model, []float32, decoderOracle) {
	b.Helper()
	path := os.Getenv("GOPHONIC_WHISPER_MODEL")
	if path == "" {
		b.Skip("set GOPHONIC_WHISPER_MODEL to the converted official tiny.en weights")
	}
	m, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	dir := filepath.Join("..", "testdata", "whisper")
	encoder, err := readF32Fixture(filepath.Join(dir, "jfk.encoder.f32le"), AudioFrames*AudioState)
	if err != nil {
		b.Fatal(err)
	}
	manifest, err := os.ReadFile(filepath.Join(dir, "jfk.oracle.json"))
	if err != nil {
		b.Fatal(err)
	}
	var oracle decoderOracle
	if err := json.Unmarshal(manifest, &oracle); err != nil {
		b.Fatal(err)
	}
	return m, encoder, oracle
}

func BenchmarkDecoderOfficialGreedy(b *testing.B) {
	benchmarkDecoderOfficialGreedy(b, 0)
}

func BenchmarkDecoderOfficialGreedyWorkers(b *testing.B) {
	for _, workers := range []int{1, 8} {
		b.Run("workers_"+strconv.Itoa(workers), func(b *testing.B) {
			benchmarkDecoderOfficialGreedy(b, workers)
		})
	}
}

func benchmarkDecoderOfficialGreedy(b *testing.B, workers int) {
	m, encoder, oracle := decoderBenchmarkFixture(b)
	s := NewDecoderScratch()
	if workers != 0 {
		pool, err := whispergemm.NewExecutor(workers)
		if err != nil {
			b.Fatal(err)
		}
		defer pool.Close()
		s.gemm = pool
	}
	output := make([]int, len(oracle.Prefix)+len(oracle.Tokens)+1)
	n, err := m.GreedyDecodeInto(encoder, oracle.Prefix, output, s, 50256)
	if err != nil {
		b.Fatal(err)
	}
	if n != len(output) || output[n-1] != 50256 {
		b.Fatalf("greedy benchmark must produce complete official output: %v", output[:n])
	}
	for i, want := range oracle.Tokens {
		if output[len(oracle.Prefix)+i] != want {
			b.Fatalf("greedy token[%d]=%d want %d", i, output[len(oracle.Prefix)+i], want)
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		n, err = m.GreedyDecodeInto(encoder, oracle.Prefix, output, s, 50256)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(oracle.Tokens)), "tokens/op")
	decoderBenchmarkSink = output[n-1]
}

func BenchmarkDecoderOfficialBegin(b *testing.B) {
	m, encoder, _ := decoderBenchmarkFixture(b)
	s := NewDecoderScratch()
	if err := m.BeginDecode(encoder, s); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if err := m.BeginDecode(encoder, s); err != nil {
			b.Fatal(err)
		}
	}
	decoderBenchmarkSink = s.audioFrames
}
