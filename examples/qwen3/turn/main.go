// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command turn decides, word by word, whether a voice-agent user has
// finished their turn: the "should I talk now?" call. Each utterance is fed
// as growing partial transcripts, as a speech recognizer emits them, and a
// Stream evaluates only the words added since the previous partial.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/GetStream/gophonic/qwen3"
)

const question = "A voice assistant hears this live, unpunctuated transcript. Has the user finished their turn, or did they stop mid-sentence and will keep talking?"

var utterances = []string{
	"can you book me a table for two at seven tonight",
	"so i was thinking maybe we could",
	"what's the weather like in lisbon tomorrow",
	"no thanks that's everything",
}

func main() {
	m, err := qwen3.Open(os.Getenv("GOPHONIC_QWEN3_MODEL"), qwen3.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	q, err := m.Question(question, []string{"reply now", "wait"})
	if err != nil {
		log.Fatal(err)
	}
	probs := make([]float32, 2)
	for _, u := range utterances {
		s, err := q.NewStream(128)
		if err != nil {
			log.Fatal(err)
		}
		words := strings.Fields(u)
		for i := range words {
			partial := strings.Join(words[:i+1], " ")
			start := time.Now()
			if err := s.Update(context.Background(), partial, probs); err != nil {
				log.Fatal(err)
			}
			verdict := "wait"
			if probs[0] > 0.5 {
				verdict = "reply now"
			}
			fmt.Printf("%-9s %.2f %4dms  %s\n", verdict, probs[0], time.Since(start).Milliseconds(), partial)
		}
		fmt.Println()
	}
}
