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

// TestOfficialExamples runs the reply example end to end on the official
// checkpoint and checks that the helpful reply ranks first in every
// conversation.
func TestOfficialExamples(t *testing.T) {
	if os.Getenv("GOPHONIC_QWEN3_MODEL") == "" || os.Getenv("GOPHONIC_CLM_HEAD_BUNDLE") == "" {
		t.Skip("set GOPHONIC_QWEN3_MODEL and GOPHONIC_CLM_HEAD_BUNDLE to run the examples")
	}
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go tool not found")
	}
	cmd := exec.CommandContext(t.Context(), goTool, "run", "./reply")
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Logf("\n%s", out)
	// Each conversation prints its state, then candidate lines in input
	// order; the first candidate is the helpful one and must score highest.
	var best, first float64
	var inBlock bool
	check := func() {
		if inBlock && first < best {
			t.Errorf("a distractor outscored the helpful reply (%.2f < %.2f)", first, best)
		}
	}
	idx := 0
	for line := range strings.Lines(string(out)) {
		if !strings.HasPrefix(line, "  ") {
			check()
			inBlock, best, idx = strings.Contains(line, ": "), 0, 0
			continue
		}
		p, err := strconv.ParseFloat(strings.Fields(line)[0], 64)
		if err != nil {
			t.Fatalf("parse %q: %v", line, err)
		}
		if idx == 0 {
			first = p
		}
		best = max(best, p)
		idx++
	}
	check()
}
