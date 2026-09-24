// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command sentiment labels the emotion of short texts, including sarcasm.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"slices"

	"github.com/GetStream/gophonic/qwen3"
)

const question = "What emotion does the writer express?"

var options = []string{"joy", "anger or frustration", "sadness", "fear", "neutral"}

var inputs = []string{
	"Honestly the best pizza I've had in years!",
	"I waited two hours and nobody even apologized.",
	"I can't believe she's gone. I keep reaching for my phone to call her.",
	"Is anyone else hearing footsteps upstairs? I live alone.",
	"The package arrived on Tuesday.",
	"Oh great, another Monday. Just what I needed.",
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
