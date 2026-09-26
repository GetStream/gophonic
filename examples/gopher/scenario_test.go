// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/duplex"
	"github.com/GetStream/gophonic/scenario"
	"github.com/GetStream/gophonic/speech"
)

// TestScenarios runs Gopher, as config makes it, through the scripts in
// scenarios/: each line the user says is spoken by Qwen3-TTS in another
// voice, Gopher's answers are transcribed by Qwen3-ASR, and the language
// model judges what they say. It needs the three models (GOPHONIC_MODELS,
// or models/ at the repository root; GOPHER_LLM names another language
// model directory there) and runs in real time.
func TestScenarios(t *testing.T) {
	if testing.Short() {
		t.Skip("scenarios run the models in real time")
	}
	dir := os.Getenv("GOPHONIC_MODELS")
	if dir == "" {
		dir = filepath.Join("..", "..", "models")
	}
	open := func(name string) *gophonic.Model {
		path := filepath.Join(dir, name)
		if _, err := os.Stat(path); err != nil {
			t.Skipf("model %s is not in %s", name, dir)
		}
		m, err := gophonic.Open(path, gophonic.Options{})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	llmName := os.Getenv("GOPHER_LLM")
	if llmName == "" {
		llmName = "Qwen3.6-35B-A3B"
	}
	asr, llm, tts := open("Qwen3-ASR-1.7B"), open(llmName), open("Qwen3-TTS-12Hz-1.7B-CustomVoice")
	lane := func(t *testing.T, l interface{ Close() error }, err error) {
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { l.Close() })
	}
	ears, err := gophonic.Lane[speech.Transcriber](asr)
	lane(t, ears, err)
	voice, err := gophonic.Lane[speech.Synthesizer](tts)
	lane(t, voice, err)
	judge, err := gophonic.Lane[chat.Generator](llm)
	lane(t, judge, err)
	// Each script gets a fresh Gopher, as config makes it, so that no
	// conversation leaks from one into the next.
	setup := func(t *testing.T) scenario.Config {
		captions := &scenario.Captions{}
		cfg := config("aiden", "", []string{"en", "pt"})
		cfg.Observer = captions
		agent, err := duplex.New(cfg, asr, llm, tts)
		lane(t, agent, err)
		return scenario.Config{Agent: agent, Voice: voice, Speak: speech.SpeakOptions{Voice: "serena"},
			Ears: ears, Listen: cfg.Listen, Judge: judge, Captions: captions}
	}
	scenario.Test(t, setup, "scenarios/*.txt")
}
