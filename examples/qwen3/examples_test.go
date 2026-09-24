// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package examples

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestOfficialExamples runs every example on the official checkpoint and
// checks its answers. Each example loads the model itself, so they run one
// after another.
func TestOfficialExamples(t *testing.T) {
	if os.Getenv("GOPHONIC_QWEN3_MODEL") == "" || os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE") == "" {
		t.Skip("set GOPHONIC_QWEN3_MODEL and GOPHONIC_CLM_HEAD_BUNDLE to run the examples")
	}
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go tool not found")
	}
	// want lists each output line's expected first field, in order.
	for example, want := range map[string][]string{
		"intent":    {"payments", "cancellations", "technical", "shipping", "account"},
		"subtitles": {"horror", "news", "sports", "cooking", "romance", "science"},
		"sentiment": {"joy", "anger", "sadness", "", "neutral", "anger"}, // "" = not checked
		"tools":     {"search_flights", "get_weather", "book_restaurant", "", "set_timer", "play_music"},
	} {
		t.Run(example, func(t *testing.T) {
			out, err := exec.CommandContext(t.Context(), goTool, "run", "./"+example).CombinedOutput()
			if err != nil {
				t.Fatalf("%v\n%s", err, out)
			}
			t.Logf("\n%s", out)
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			if len(lines) != len(want) {
				t.Fatalf("got %d lines, want %d", len(lines), len(want))
			}
			for i, w := range want {
				if got := strings.Fields(lines[i])[0]; w != "" && got != w {
					t.Errorf("line %d: got %q, want %q", i+1, got, w)
				}
			}
		})
	}
	t.Run("turn", func(t *testing.T) {
		out, err := exec.CommandContext(t.Context(), goTool, "run", "./turn").CombinedOutput()
		if err != nil {
			t.Fatalf("%v\n%s", err, out)
		}
		t.Logf("\n%s", out)
		// The last partial of each utterance is the complete sentence.
		want := []string{"reply", "wait", "reply", "reply"}
		groups := strings.Split(strings.TrimSpace(string(out)), "\n\n")
		if len(groups) != len(want) {
			t.Fatalf("got %d utterances, want %d", len(groups), len(want))
		}
		for i, g := range groups {
			lines := strings.Split(g, "\n")
			if got := strings.Fields(lines[len(lines)-1])[0]; got != want[i] {
				t.Errorf("utterance %d ends with %q, want %q", i+1, got, want[i])
			}
		}
	})
	t.Run("conversation", func(t *testing.T) {
		out, err := exec.CommandContext(t.Context(), goTool, "run", "./conversation").CombinedOutput()
		if err != nil {
			t.Fatalf("%v\n%s", err, out)
		}
		t.Logf("\n%s", out)
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		first, last := lines[1], lines[len(lines)-1]
		for _, w := range []string{"mood=angry", "resolved=no"} {
			if !strings.Contains(first, w) {
				t.Errorf("after the first turn: %q lacks %s", first, w)
			}
		}
		for _, w := range []string{"mood=satisfied", "topic=billing", "resolved=yes"} {
			if !strings.Contains(last, w) {
				t.Errorf("after the last turn: %q lacks %s", last, w)
			}
		}
	})
	t.Run("reply", func(t *testing.T) {
		out, err := exec.CommandContext(t.Context(), goTool, "run", "./reply").CombinedOutput()
		if err != nil {
			t.Fatalf("%v\n%s", err, out)
		}
		t.Logf("\n%s", out)
		// Rank sorts candidates best first, so the line after each
		// conversation must be its helpful reply.
		helpful := []string{"I'm sorry about the double charge", "I'm so sorry. You can rebook", "slices.Concat was added"}
		var firsts []string
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		for i, line := range lines {
			if !strings.HasPrefix(line, "  ") && i+1 < len(lines) {
				firsts = append(firsts, lines[i+1])
			}
		}
		if len(firsts) != len(helpful) {
			t.Fatalf("got %d conversations, want %d", len(firsts), len(helpful))
		}
		for i, h := range helpful {
			if !strings.Contains(firsts[i], h) {
				t.Errorf("conversation %d ranked %q first, want %q", i+1, strings.TrimSpace(firsts[i]), h)
			}
		}
	})
}
