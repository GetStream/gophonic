// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command intent routes support messages to the right team.
package main

import (
	"fmt"
	"log"

	"github.com/GetStream/gophonic/examples/clm-qwen/examples/internal/demo"
)

var (
	teams      = []string{"billing", "cancel", "tech", "shipping", "login"}
	candidates = []string{
		"Transfer to the billing and payments team.",
		"Transfer to the cancellations team.",
		"Transfer to technical support.",
		"Transfer to the shipping and delivery team.",
		"Transfer to account login and security support.",
	}
)

var messages = []string{
	"my card got declined at the gas station even though I have money",
	"cancel my plan before it renews next week",
	"the app crashes every time I open settings",
	"where's my package, it says delivered but it's not here",
	"I forgot my password and the reset email never arrives",
}

func main() {
	m, err := demo.Open()
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	for _, msg := range messages {
		p, err := m.Probs("Customer: "+msg, candidates)
		if err != nil {
			log.Fatal(err)
		}
		best := demo.Best(p)
		fmt.Printf("%-8s %.2f  %q\n", teams[best], p[best], msg)
	}
	fmt.Println(m.Summary())
}
