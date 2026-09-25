// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command subtitles guesses what kind of program each subtitle line comes from.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"slices"

	"github.com/GetStream/gophonic/qwen3"
)

const question = "What kind of program is this subtitle line most likely from?"

var options = []string{"horror", "news", "sports", "cooking", "romance", "science fiction"}

var inputs = []string{
	"Don't go down there. Whatever you do, don't open that door.",
	"The central bank raised interest rates by half a point this afternoon.",
	"He shoots... he scores! What a goal in the ninetieth minute!",
	"Now add a pinch of salt and let the onions caramelize for ten minutes.",
	"I've loved you since the first day I saw you in that bookshop.",
	"Captain, the warp core is going critical. We have ninety seconds.",
}

func main() {
	modelPath := flag.String("model", "models/Qwen3-8B", "official Qwen3-8B safetensors directory")
	flag.Parse()
	m, err := qwen3.Open(*modelPath, qwen3.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	q, err := m.Question(question, options) // tokenizes the fixed prompt once
	if err != nil {
		log.Fatal(err)
	}
	// Classify every cue in one batch: inputs share forward passes.
	probs := make([][]float32, len(inputs))
	for i := range probs {
		probs[i] = make([]float32, len(options))
	}
	if err := q.ChooseBatch(context.Background(), inputs, probs); err != nil {
		log.Fatal(err)
	}
	for i, in := range inputs {
		best := slices.Index(probs[i], slices.Max(probs[i]))
		fmt.Printf("%-20s %.2f  %s\n", options[best], probs[i][best], in)
	}
}
