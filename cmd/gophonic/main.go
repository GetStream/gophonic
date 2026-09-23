// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/GetStream/gophonic"
)

func main() {
	modelPath := flag.String("model", "", "converted Pipecat Smart Turn v3.2 .gophonic bundle")
	tinyModelPath := flag.String("tiny-model", "", "converted TinyMelNet .gophonic bundle (uses its own threshold)")
	tinyWorkers := flag.Int("tiny-workers", 0, "TinyMelNet persistent CPU helpers (0=serial, max 7; only with -tiny-model)")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: gophonic (-model smart-turn.gophonic | -tiny-model tinymel.gophonic [-tiny-workers 0..7]) audio.wav|audio.ogg|audio.opus\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if (*modelPath == "") == (*tinyModelPath == "") || flag.NArg() != 1 || *tinyWorkers < 0 || *tinyWorkers > gophonic.MaxTinyMelWorkers || (*modelPath != "" && *tinyWorkers != 0) {
		flag.Usage()
		os.Exit(2)
	}
	pcm, rate, channels, err := readAudio(flag.Arg(0))
	if err != nil {
		fatal(err)
	}
	var prediction gophonic.Prediction
	if *tinyModelPath != "" {
		model, err := gophonic.LoadTinyMel(*tinyModelPath)
		if err != nil {
			fatal(err)
		}
		workspace := gophonic.NewTinyMelWorkspaceWithWorkers(*tinyWorkers)
		defer workspace.Close()
		prediction, err = model.PredictInto(pcm, rate, channels, workspace)
	} else {
		model, err := gophonic.Load(*modelPath)
		if err != nil {
			fatal(err)
		}
		workspace := gophonic.NewWorkspace()
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
	fmt.Fprintln(os.Stderr, "gophonic:", err)
	os.Exit(1)
}
