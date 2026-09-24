// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command tools picks the tool an assistant should call for each request.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"slices"

	"github.com/GetStream/gophonic/qwen3"
)

const question = "Which tool should the assistant call for this request?"

var options = []string{"search_flights", "get_weather", "book_restaurant", "send_message", "set_timer", "play_music"}

var inputs = []string{
	"Find me something from Lisbon to Berlin next Friday.",
	"Will I need an umbrella in Porto tomorrow?",
	"Get us a table for four at Nobu at 8pm.",
	"Tell Maria I'll be ten minutes late to the standup.",
	"Remind me in 12 minutes to take the pasta out.",
	"Put on some Nina Simone.",
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
