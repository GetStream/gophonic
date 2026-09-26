// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// benchmark_whisper_go_warm measures repeated 30-second-window PCM-to-text
// inference with a loaded model and reusable transcriber. It prints one
// nanosecond duration per timed call after five warm calls, matching the
// 24-token JFK output of benchmark_whisper_cpp_warm.cpp.
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/gophonic/whisper"
)

const expected = "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."

func main() {
	if len(os.Args) != 5 && len(os.Args) != 6 {
		fatal("usage: benchmark_whisper_go_warm model pcm-f32le threads iterations [expected-text]")
	}
	want := expected
	if len(os.Args) == 6 {
		want = os.Args[5]
	}
	threads, err := strconv.Atoi(os.Args[3])
	if err != nil || threads < 1 {
		fatal("threads must be positive")
	}
	iterations, err := strconv.Atoi(os.Args[4])
	if err != nil || iterations < 1 {
		fatal("iterations must be positive")
	}
	runtime.GOMAXPROCS(threads)
	model, err := whisper.Load(os.Args[1])
	if err != nil {
		fatal(err.Error())
	}
	defer model.Close()
	worker, err := whisper.NewTranscriber(model, whisper.LaneOptions{Threads: threads})
	if err != nil {
		fatal(err.Error())
	}
	defer worker.Close()
	data, err := os.ReadFile(os.Args[2])
	if err != nil || len(data) == 0 || len(data)%4 != 0 {
		fatal("invalid PCM input")
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	output := make([]byte, 0, 2048)
	for i := -5; i < iterations; i++ {
		start := time.Now()
		text, err := worker.TranscribeWindowInto(pcm, output)
		duration := time.Since(start)
		if err != nil {
			fatal(err.Error())
		}
		if strings.TrimSpace(string(text)) != want {
			fatal("transcript differs from expected text: " + strings.TrimSpace(string(text)))
		}
		if i >= 0 {
			fmt.Println(duration.Nanoseconds())
		}
	}
}

func fatal(message string) {
	fmt.Fprintln(os.Stderr, message)
	os.Exit(2)
}
