// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command intent routes customer messages to a support team.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"slices"

	"github.com/GetStream/gophonic/qwen3"
)

const question = "Which support team should handle this customer message?"

var options = []string{"payments", "cancellations", "technical support", "shipping", "account login"}

var inputs = []string{
	"my card got declined at the gas station even though I have money",
	"cancel my plan before it renews next week",
	"the app crashes every time I open settings",
	"where's my package, it says delivered but it's not here",
	"I forgot my password and the reset email never arrives",
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
