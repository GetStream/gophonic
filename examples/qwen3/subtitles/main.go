// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command subtitles guesses what kind of program each subtitle line comes from.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
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
	m, err := qwen3.Open(os.Getenv("GOPHONIC_QWEN3_MODEL"), qwen3.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	q, err := m.Question(question, options) // tokenizes the fixed prompt once
	if err != nil {
		log.Fatal(err)
	}
	probs := make([]float32, len(options))
	for _, in := range inputs {
		if err := q.Choose(context.Background(), in, probs); err != nil {
			log.Fatal(err)
		}
		best := slices.Index(probs, slices.Max(probs))
		fmt.Printf("%-20s %.2f  %s\n", options[best], probs[best], in)
	}
}
