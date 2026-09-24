// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command reply picks the best agent reply for a conversation, the task CLM
// was trained for: ranking candidate actions for a state.
package main

import (
	"fmt"
	"log"

	"github.com/GetStream/gophonic/examples/clm-qwen/examples/internal/demo"
)

var conversations = []struct {
	state   string
	replies []string
}{
	{
		"Customer: I was charged twice for order #4411 and I want my money back today.",
		[]string{
			"I'm sorry about the double charge. I've refunded the duplicate payment for order #4411; it will reach your card in 3-5 business days.",
			"Have you tried restarting your device?",
			"Thanks for the kind words! We love hearing from happy customers.",
			"Refunds are not possible.",
		},
	},
	{
		"User: my flight got cancelled and I'm stuck in Frankfurt with two kids. what are my options?",
		[]string{
			"I'm so sorry. You can rebook on the next flight at no cost, and because you're stranded overnight the airline owes you a hotel and meals. Shall I look up the next departures?",
			"Frankfurt has a beautiful old town you could visit.",
			"Please contact your airline.",
			"Here is a recipe for apple strudel.",
		},
	},
	{
		"User: the build fails with 'undefined: slices.Concat' on our CI but works on my laptop.",
		[]string{
			"slices.Concat was added in Go 1.22; your CI is likely on an older Go. Check `go version` there and bump the toolchain or the go directive.",
			"Have you tried turning it off and on again?",
			"Go is a statically typed language developed at Google.",
			"Please open a ticket with IT.",
		},
	},
}

func main() {
	m, err := demo.Open()
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	for _, c := range conversations {
		p, err := m.Probs(c.state, c.replies)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(c.state)
		for i, r := range c.replies {
			fmt.Printf("  %.2f  %.70s\n", p[i], r)
		}
	}
	fmt.Println(m.Summary())
}
