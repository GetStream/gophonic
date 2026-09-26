// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command qwen3-gptq rounds a Qwen3-8B checkpoint's GPU weights with GPTQ
// once, next to the checkpoint; qwen3.Open then loads them automatically.
//
//	qwen3-gptq /path/to/Qwen3-8B           # int8 (the GPU default)
//	qwen3-gptq -weights gpu-q4 /path/to/Qwen3-8B
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/GetStream/gophonic/qwen3"
)

func main() {
	weights := flag.String("weights", "gpu", "weight format: gpu or gpu-q4")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: qwen3-gptq [-weights gpu|gpu-q4] /path/to/Qwen3-8B")
		os.Exit(2)
	}
	start := time.Now()
	err := qwen3.QuantizeGPTQ(flag.Arg(0), *weights, func(layer, layers int) {
		fmt.Fprintf(os.Stderr, "\rlayer %d/%d  %v", layer, layers, time.Since(start).Round(time.Second))
	})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		log.Fatal(err)
	}
}
