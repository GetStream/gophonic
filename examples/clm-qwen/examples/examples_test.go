// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package examples

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

// TestOfficialExamples runs the turn and sentiment examples end to end on the
// official checkpoint and checks their clear-cut verdicts. Each example loads
// the model itself, so they run one after the other.
func TestOfficialExamples(t *testing.T) {
	if os.Getenv("GOPHONIC_QWEN3_MODEL") == "" || os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE") == "" {
		t.Skip("set GOPHONIC_QWEN3_MODEL and GOPHONIC_CLM_HEAD_BUNDLE to run the examples")
	}
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go tool not found")
	}
	tests := []struct {
		example string
		want    map[string]string // input text -> first output field
	}{
		{"turn", map[string]string{
			"can you book me a table for two at seven tonight": "reply",
			"i'd like to fly to":          "wait",
			"no thanks that's everything": "reply",
		}},
		{"sentiment", map[string]string{
			"Honestly the best pizza I've had in years!":                            "joy",
			"I waited two hours and nobody even apologized.":                        "anger",
			"I can't believe she's gone. I keep reaching for my phone to call her.": "sadness",
		}},
	}
	for _, tt := range tests {
		t.Run(tt.example, func(t *testing.T) {
			cmd := exec.CommandContext(t.Context(), goTool, "run", "./"+tt.example)
			cmd.Stderr = os.Stderr
			out, err := cmd.Output()
			if err != nil {
				t.Fatalf("%v\n%s", err, out)
			}
			t.Logf("\n%s", out)
			for text, want := range tt.want {
				line := findLine(string(out), strconv.Quote(text))
				if line == "" {
					t.Errorf("no output line for %q", text)
				} else if got := strings.Fields(line)[0]; got != want {
					t.Errorf("%q: got %s, want %s", text, got, want)
				}
			}
		})
	}
}

func findLine(out, quoted string) string {
	for line := range strings.Lines(out) {
		if strings.HasSuffix(strings.TrimSpace(line), quoted) {
			return line
		}
	}
	return ""
}
