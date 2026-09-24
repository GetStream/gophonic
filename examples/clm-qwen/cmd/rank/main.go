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
	clmqwen "github.com/GetStream/gophonic/examples/clm-qwen"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	qwenPath := flag.String("qwen", "", "official Qwen3-8B safetensors directory or GGUF path")
	headPath := flag.String("head", "", "converted CLM .gclm head bundle")
	state := flag.String("state", "", "state text to rank candidate actions against")
	quant := flag.String("quant", "", "Qwen CPU quantization: empty (FP32) or int8")
	flag.Parse()
	if *qwenPath == "" || *headPath == "" || *state == "" || flag.NArg() == 0 {
		return errors.New("usage: rank -qwen PATH -head PATH -state TEXT [-quant int8] CANDIDATE...")
	}
	head, err := clm.Load(*headPath)
	if err != nil {
		return fmt.Errorf("load CLM head: %w", err)
	}
	encoder, err := clmqwen.OpenWithOptions(*qwenPath, clmqwen.Options{Quant: *quant})
	if err != nil {
		return fmt.Errorf("load Qwen3: %w", err)
	}
	defer encoder.Close()
	engine, err := clm.NewEngine(head, encoder)
	if err != nil {
		return err
	}
	results, err := engine.Rank(context.Background(), *state, flag.Args(), 1)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(results)
}
