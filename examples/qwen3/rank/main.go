// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command rank runs CLM-v0.1-8B locally with a Qwen3-8B CPU backbone.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/GetStream/gophonic/clm"
	"github.com/GetStream/gophonic/qwen3"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	qwenPath := flag.String("qwen", "", "official Qwen3-8B safetensors directory")
	headPath := flag.String("head", "", "converted CLM .gclm head bundle")
	state := flag.String("state", "", "state text to rank candidate actions against")
	weights := flag.String("weights", "", "projection weights: f16 (exact BF16, default) or int8")
	threads := flag.Int("threads", 0, "worker threads (0 selects a default)")
	flag.Parse()
	if *qwenPath == "" || *headPath == "" || *state == "" || flag.NArg() == 0 {
		return errors.New("usage: rank -qwen PATH -head PATH -state TEXT [-weights int8] CANDIDATE...")
	}
	head, err := clm.Load(*headPath)
	if err != nil {
		return fmt.Errorf("load CLM head: %w", err)
	}
	m, err := qwen3.Open(*qwenPath, qwen3.Options{Weights: *weights, Threads: *threads})
	if err != nil {
		return fmt.Errorf("load Qwen3: %w", err)
	}
	defer m.Close()
	engine, err := clm.NewEngine(head, clm.EmbedFunc(func(ctx context.Context, _ clm.Role, texts []string, dst [][]float32) error {
		return m.Embed(ctx, texts, dst)
	}))
	if err != nil {
		return err
	}
	results, err := engine.Rank(context.Background(), *state, flag.Args(), 1)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(results)
}
