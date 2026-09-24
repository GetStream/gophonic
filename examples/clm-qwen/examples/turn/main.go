// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command turn decides whether a voice-agent user has finished speaking,
// from an unpunctuated live transcript: the "should I talk now?" call.
package main

import (
	"fmt"
	"log"

	"github.com/GetStream/gophonic/examples/clm-qwen/examples/internal/demo"
)

var candidates = []string{
	"The user has finished speaking and is waiting for a reply.",
	"The user is in the middle of a sentence and is about to say more.",
}

var transcripts = []string{
	"can you book me a table for two at seven tonight",
	"so i was thinking maybe we could",
	"what's the weather like in lisbon tomorrow",
	"i'd like to fly to",
	"no thanks that's everything",
	"the problem is that when i open the app it",
}

func main() {
	m, err := demo.Open()
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	for _, text := range transcripts {
		p, err := m.Probs(text, candidates)
		if err != nil {
			log.Fatal(err)
		}
		verdict := "wait"
		if p[0] > p[1] {
			verdict = "reply"
		}
		fmt.Printf("%-5s  P(done)=%.2f  %q\n", verdict, p[0], text)
	}
	fmt.Println(m.Summary())
}
