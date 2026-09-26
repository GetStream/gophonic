// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/internal/whispergemm"
	"github.com/thesyncim/vibejson"
)

var decoderBenchmarkSink int

func decoderBenchmarkFixture(b *testing.B) (*Model, []float32, decoderOracle) {
	b.Helper()
	path := testmodels.Path(b, testmodels.WhisperTinyEN)
	m, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	dir := filepath.Join("..", "testdata", "whisper")
	encoder, err := readF32Fixture(filepath.Join(dir, "jfk.encoder.f32le"), audioFrames*audioState)
	if err != nil {
		b.Fatal(err)
	}
	manifest, err := os.ReadFile(filepath.Join(dir, "jfk.oracle.json"))
	if err != nil {
		b.Fatal(err)
	}
	var oracle decoderOracle
	if err := vibejson.Unmarshal(manifest, &oracle); err != nil {
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
	s := newTinyDecoderScratch()
	if workers != 0 {
		pool, err := whispergemm.NewExecutor(workers)
		if err != nil {
			b.Fatal(err)
		}
		defer pool.Close()
		s.gemm = pool
	}
	output := make([]int, len(oracle.Prefix)+len(oracle.Tokens)+1)
	n, err := m.greedyDecodeInto(encoder, oracle.Prefix, output, s, 50256)
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
		n, err = m.greedyDecodeInto(encoder, oracle.Prefix, output, s, 50256)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(oracle.Tokens)), "tokens/op")
	decoderBenchmarkSink = output[n-1]
}

func BenchmarkDecoderOfficialBegin(b *testing.B) {
	benchmarkDecoderOfficialBegin(b, 0)
}

func BenchmarkDecoderOfficialBeginWorkers(b *testing.B) {
	for _, workers := range []int{1, 8} {
		b.Run("workers_"+strconv.Itoa(workers), func(b *testing.B) {
			benchmarkDecoderOfficialBegin(b, workers)
		})
	}
}

func benchmarkDecoderOfficialBegin(b *testing.B, workers int) {
	m, encoder, _ := decoderBenchmarkFixture(b)
	s := newTinyDecoderScratch()
	if workers != 0 {
		pool, err := whispergemm.NewExecutor(workers)
		if err != nil {
			b.Fatal(err)
		}
		defer pool.Close()
		s.gemm = pool
	}
	if err := m.beginDecode(encoder, s); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if err := m.beginDecode(encoder, s); err != nil {
			b.Fatal(err)
		}
	}
	decoderBenchmarkSink = s.audioFrames
}

// This isolates incremental token work from BeginDecode's cross-KV projection.
// whisper.cpp charges cross-KV projection to its encoder timing, so compare
// total transcription or matched stage boundaries rather than decoder labels.
func BenchmarkDecoderOfficialTokensWorkers(b *testing.B) {
	for _, workers := range []int{1, 8} {
		b.Run("workers_"+strconv.Itoa(workers), func(b *testing.B) {
			m, encoder, oracle := decoderBenchmarkFixture(b)
			s := newTinyDecoderScratch()
			pool, err := whispergemm.NewExecutor(workers)
			if err != nil {
				b.Fatal(err)
			}
			defer pool.Close()
			s.gemm = pool
			output := make([]int, len(oracle.Prefix)+len(oracle.Tokens)+1)
			n, err := m.greedyDecodeInto(encoder, oracle.Prefix, output, s, 50256)
			if err != nil {
				b.Fatal(err)
			}
			for i, want := range oracle.Tokens {
				if output[len(oracle.Prefix)+i] != want {
					b.Fatalf("greedy token[%d]=%d want %d", i, output[len(oracle.Prefix)+i], want)
				}
			}
			if n != len(output) || output[n-1] != 50256 {
				b.Fatalf("greedy benchmark must produce complete official output: %v", output[:n])
			}
			b.ReportAllocs()
			for b.Loop() {
				s.nextPos = 0
				for position := range output[:n-1] {
					if err := m.logitsForTokenInto(output[position], position, s, s.logits); err != nil {
						b.Fatal(err)
					}
					if position+1 >= len(oracle.Prefix) {
						output[position+1] = argmax(s.logits)
					}
				}
			}
			b.ReportMetric(float64(len(oracle.Tokens)), "tokens/op")
			decoderBenchmarkSink = output[n-1]
		})
	}
}
