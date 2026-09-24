// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command sentiment labels the emotion behind short texts, sarcasm included.
package main

import (
	"fmt"
	"log"

	"github.com/GetStream/gophonic/examples/clm-qwen/examples/internal/demo"
)

var (
	names      = []string{"joy", "anger", "sadness", "fear", "neutral"}
	candidates = []string{
		"The writer feels happy.",
		"The writer feels angry.",
		"The writer feels sad.",
		"The writer feels afraid.",
		"The writer is neutral, just stating a fact.",
	}
)

var texts = []string{
	"Honestly the best pizza I've had in years!",
	"I waited two hours and nobody even apologized.",
	"I can't believe she's gone. I keep reaching for my phone to call her.",
	"Is anyone else hearing footsteps upstairs? I live alone.",
	"The package arrived on Tuesday.",
	"Oh great, another Monday. Just what I needed.",
}

func main() {
	m, err := demo.Open()
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	for _, text := range texts {
		p, err := m.Probs(text, candidates)
		if err != nil {
			log.Fatal(err)
		}
		best := demo.Best(p)
		fmt.Printf("%-8s %.2f  %q\n", names[best], p[best], text)
	}
	fmt.Println(m.Summary())
}
