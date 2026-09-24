// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command turn decides whether a voice-agent user has finished their turn from a
// live, unpunctuated transcript: the "should I talk now?" call.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"slices"

	"github.com/GetStream/gophonic/qwen3"
)

const question = "A voice assistant hears this live, unpunctuated transcript. Has the user finished their turn, or did they stop mid-sentence and will keep talking?"

var options = []string{"reply now", "wait"}

var inputs = []string{
	"can you book me a table for two at seven tonight",
	"so i was thinking maybe we could",
	"what's the weather like in lisbon tomorrow",
	"i'd like to fly to",
	"no thanks that's everything",
	"the problem is that when i open the app it",
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
