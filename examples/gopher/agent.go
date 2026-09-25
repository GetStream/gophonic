// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/duplex"
	"github.com/GetStream/gophonic/speech"
)

const prompt = `You are Gopher, a friendly voice assistant taking part in a live video call.
Everything you write is spoken aloud, so answer in one to three short, natural sentences.
Never use emoji, symbols, lists, or markdown. You run entirely on the user's own laptop, in Go.
Today is %s, in the %s time zone. Your knowledge may be older than that: when someone tells you about something
newer, believe them rather than insisting on what you knew.
In a meeting, what people say reaches you as "name said: ..." and you answer only what is meant
for you. Notes about the call reach you as system messages: who joins or leaves, and what people
type in the call's chat. Use them when asked, to repeat, spell, or summarize what someone wrote,
but never read a chat message out loud or answer it unless someone asks you to.
Use your tools rather than guessing: for the time, call now; for facts you are unsure of, or that
may have changed, call search and answer from what it finds.`

// config is Gopher: its prompt, voice, languages, and tools. main adds how
// it names the people in the call and shows its captions; the scenario
// tests run it as it is.
func config(voice, language string, languages []string) duplex.Config {
	system := fmt.Sprintf(prompt, time.Now().Format("Monday, January 2, 2006"), localZone())
	if len(languages) > 0 {
		var names []string
		for _, code := range languages {
			if name, ok := speech.LanguageName(code); ok {
				names = append(names, name)
			}
		}
		system += "\nPeople in this call speak " + strings.Join(names, " and ") +
			". Answer in the one you are spoken to in, and never in another."
	}
	return duplex.Config{
		Prompt: system,
		Voice:  speech.SpeakOptions{Voice: voice, Language: language},
		Listen: speech.Options{Language: language, Languages: languages},
		Reply:  chat.Options{Temperature: 0.7, TopP: 0.9, MaxTokens: 160},
		Tools:  tools(),
	}
}
