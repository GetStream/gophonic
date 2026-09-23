// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/GetStream/gofloor"
)

func main() {
	modelPath := flag.String("model", "", "converted Pipecat Smart Turn v3.2 .gofloor bundle")
	tinyModelPath := flag.String("tiny-model", "", "converted TinyMelNet .gofloor bundle (uses its own threshold)")
	tinyWorkers := flag.Int("tiny-workers", 0, "TinyMelNet persistent CPU helpers (0=serial, max 7; only with -tiny-model)")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: gofloor (-model smart-turn.gofloor | -tiny-model tinymel.gofloor [-tiny-workers 0..7]) audio.wav|audio.ogg|audio.opus\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if (*modelPath == "") == (*tinyModelPath == "") || flag.NArg() != 1 || *tinyWorkers < 0 || *tinyWorkers > gofloor.MaxTinyMelWorkers || (*modelPath != "" && *tinyWorkers != 0) {
		flag.Usage()
		os.Exit(2)
	}
	pcm, rate, channels, err := readAudio(flag.Arg(0))
	if err != nil {
		fatal(err)
	}
	var prediction gofloor.Prediction
	if *tinyModelPath != "" {
		model, err := gofloor.LoadTinyMel(*tinyModelPath)
		if err != nil {
			fatal(err)
		}
		workspace := gofloor.NewTinyMelWorkspaceWithWorkers(*tinyWorkers)
		defer workspace.Close()
		prediction, err = model.PredictInto(pcm, rate, channels, workspace)
	} else {
		model, err := gofloor.Load(*modelPath)
		if err != nil {
			fatal(err)
		}
		workspace := gofloor.NewWorkspace()
		defer workspace.Close()
		prediction, err = model.PredictInto(pcm, rate, channels, workspace)
	}
	if err != nil {
		fatal(err)
	}
	if err := json.NewEncoder(os.Stdout).Encode(prediction); err != nil {
		fatal(err)
	}
}

func readAudio(path string) ([]float32, int, int, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".wav":
		return readWAV(path)
	case ".opus", ".ogg":
		return readOggOpus(path)
	default:
		return nil, 0, 0, fmt.Errorf("unsupported audio file %q; use WAV or Ogg Opus", path)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "gofloor:", err)
	os.Exit(1)
}
