// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command rank runs CLM-v0.1-8B locally with a Qwen3-8B CPU backbone.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/GetStream/gophonic/clm"
	"github.com/GetStream/gophonic/qwen3"
	"github.com/thesyncim/vibejson"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	qwenPath := flag.String("qwen", "models/Qwen3-8B", "official Qwen3-8B safetensors directory")
	headPath := flag.String("head", "models/CLM_v0.1-8B.gclm", "converted CLM .gclm head bundle")
	state := flag.String("state", "", "state text to rank candidate actions against")
	weights := flag.String("weights", "", "projection weights: f16 (exact BF16, default) or int8")
	threads := flag.Int("threads", 0, "worker threads (0 selects a default)")
	flag.Parse()
	if *state == "" || flag.NArg() == 0 {
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
	return writeJSON(os.Stdout, &results)
}

// writeJSON streams v to out as one line of JSON.
func writeJSON[T any](out io.Writer, v *T) error {
	enc, err := vibejson.CompileEncoder[T](vibejson.EncoderOptions{})
	if err != nil {
		return err
	}
	w := vibejson.NewWriter(out)
	if err := vibejson.EncodeTo(w, enc, v); err != nil {
		return err
	}
	if err := w.Newline(); err != nil {
		return err
	}
	return w.Flush()
}
