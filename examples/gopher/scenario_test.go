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
// or models/ at the repository root) and runs in real time. GOPHER_ASR,
// GOPHER_LLM, and GOPHER_TTS name other model directories there for
// Gopher; the harness keeps the reference models to hear, speak, and judge.
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
	env := func(name, fallback string) string {
		if v := os.Getenv(name); v != "" {
			return v
		}
		return fallback
	}
	// The harness hears, speaks, and judges with the reference models,
	// whatever Gopher is made of: GOPHER_ASR, GOPHER_LLM, and GOPHER_TTS
	// choose Gopher's.
	opened := map[string]*gophonic.Model{}
	model := func(name string) *gophonic.Model {
		if opened[name] == nil {
			opened[name] = open(name)
		}
		return opened[name]
	}
	ears, voiceModel, judgeModel := model("Qwen3-ASR-1.7B"), model("Qwen3-TTS-12Hz-1.7B-CustomVoice"), model("Qwen3.6-35B-A3B")
	asr := model(env("GOPHER_ASR", "Qwen3-ASR-1.7B"))
	llm := model(env("GOPHER_LLM", "Qwen3.6-35B-A3B"))
	tts := model(env("GOPHER_TTS", "Qwen3-TTS-12Hz-1.7B-CustomVoice"))
	lane := func(t *testing.T, l interface{ Close() error }, err error) {
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { l.Close() })
	}
	hear, err := gophonic.Lane[speech.Transcriber](ears)
	lane(t, hear, err)
	voice, err := gophonic.Lane[speech.Synthesizer](voiceModel)
	lane(t, voice, err)
	judge, err := gophonic.Lane[chat.Generator](judgeModel)
	lane(t, judge, err)
	// Each script gets a fresh Gopher, as config makes it, so that no
	// conversation leaks from one into the next.
	setup := func(t *testing.T) scenario.Config {
		captions := &scenario.Captions{}
		cfg := config("aiden", speech.Unknown, speech.Languages(speech.English, speech.Portuguese))
		cfg.Observer = captions
		agent, err := duplex.New(cfg, asr, llm, tts)
		lane(t, agent, err)
		return scenario.Config{Agent: agent, Voice: voice, Speak: speech.SpeakOptions{Voice: "serena"},
			Ears: hear, Listen: cfg.Listen, Judge: judge, Captions: captions}
	}
	scenario.Test(t, setup, "scenarios/*.txt")
}
