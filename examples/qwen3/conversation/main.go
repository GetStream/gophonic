// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command conversation asks several questions about a support conversation
// as it grows. The conversation is evaluated once and extended turn by turn;
// each question only adds its own short tail.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/gophonic/qwen3"
)

var turns = []string{
	"Customer: Hi, I was charged twice for my order #4411 last week.",
	"Agent: I'm sorry about that. Let me look into the duplicate charge for you.",
	"Customer: It's been a week and nobody has answered my emails. This is really frustrating.",
	"Agent: I've now refunded the duplicate payment; it will reach your card in 3-5 business days.",
	"Customer: Okay, thanks. That's all I needed.",
}

var questions = []struct {
	name, text string
	options    []string
}{
	{"mood", "How does the customer feel right now?", []string{"satisfied", "angry", "confused"}},
	{"topic", "What is the customer's problem about?", []string{"billing", "shipping", "technical issue", "account login"}},
	{"resolved", "Is the customer's issue resolved?", []string{"yes", "no"}},
}

func main() {
	modelPath := flag.String("model", "models/Qwen3-8B", "official Qwen3-8B safetensors directory")
	flag.Parse()
	m, err := qwen3.Open(*modelPath, qwen3.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	c, err := m.NewContext(1024)
	if err != nil {
		log.Fatal(err)
	}
	qs := make([]*qwen3.ContextQuestion, len(questions))
	probs := make([][]float32, len(questions))
	for i, q := range questions {
		if qs[i], err = m.ContextQuestion(q.text, q.options); err != nil {
			log.Fatal(err)
		}
		probs[i] = make([]float32, len(q.options))
	}
	for n := 1; n <= len(turns); n++ {
		start := time.Now()
		if err := c.Set(context.Background(), strings.Join(turns[:n], "\n")); err != nil {
			log.Fatal(err)
		}
		if err := c.Ask(context.Background(), qs, probs); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s\n  ", turns[n-1])
		for i, q := range questions {
			best := slices.Index(probs[i], slices.Max(probs[i]))
			fmt.Printf("%s=%s  ", q.name, q.options[best])
		}
		fmt.Printf("(%dms)\n", time.Since(start).Milliseconds())
	}
}
