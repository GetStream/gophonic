// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Command subtitles tags each cue of a small SubRip file with the kind of
// program it most likely comes from.
package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/GetStream/gophonic/examples/clm-qwen/examples/internal/demo"
)

var (
	genres     = []string{"horror", "news", "sports", "cooking", "romance", "sci-fi", "comedy"}
	candidates = []string{
		"This is a line from a horror film.",
		"This is from a TV news broadcast.",
		"This is live sports commentary.",
		"This is from a cooking show.",
		"This is from a romantic drama.",
		"This is from a science-fiction film.",
		"This is from a stand-up comedy set.",
	}
)

const srt = `1
00:00:01,000 --> 00:00:04,000
Don't go down there.
Whatever you do, don't open that door.

2
00:00:05,000 --> 00:00:08,000
The central bank raised interest rates
by half a point this afternoon.

3
00:00:09,000 --> 00:00:11,500
He shoots... he scores!
What a goal in the ninetieth minute!

4
00:00:12,000 --> 00:00:15,000
Now add a pinch of salt and let the onions
caramelize for ten minutes.

5
00:00:16,000 --> 00:00:19,000
I've loved you since the first day
I saw you in that bookshop.

6
00:00:20,000 --> 00:00:23,000
Captain, the warp core is going critical.
We have ninety seconds.
`

func main() {
	m, err := demo.Open()
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	for _, block := range strings.Split(strings.TrimSpace(srt), "\n\n") {
		lines := strings.Split(block, "\n") // number, timing, text...
		start, _, _ := strings.Cut(lines[1], " --> ")
		text := strings.Join(lines[2:], " ")
		p, err := m.Probs(text, candidates)
		if err != nil {
			log.Fatal(err)
		}
		best := demo.Best(p)
		fmt.Printf("%s  %-7s %.2f  %s\n", start, genres[best], p[best], text)
	}
	fmt.Println(m.Summary())
}
