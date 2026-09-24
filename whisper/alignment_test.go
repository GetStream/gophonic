// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import "testing"

func TestAlignmentJumpsMonotonePath(t *testing.T) {
	worker := &Transcriber{alignMatrix: []float32{
		3, 3, 0, 0, 0,
		0, 0, 3, 3, 0,
		0, 0, 0, 0, 3,
	}}
	jumps := worker.alignmentJumps(3, 5)
	if len(jumps) != 3 || jumps[0] != 0 || jumps[1] != 1 || jumps[2] != 3 {
		t.Fatalf("alignment jumps %v, want [0 1 3]", jumps)
	}
	allocs := testing.AllocsPerRun(100, func() { worker.alignmentJumps(3, 5) })
	if allocs != 0 {
		t.Fatalf("warm DTW allocated %g objects", allocs)
	}
}

func TestRemapUTF8Boundary(t *testing.T) {
	raw := []byte{0xc3, 'x'} // incomplete two-byte sequence becomes U+FFFD
	if got := remapUTF8Boundary(raw, 1, true); got != 3 {
		t.Fatalf("end boundary %d, want 3", got)
	}
	if got := remapUTF8Boundary(raw, 1, false); got != 3 {
		t.Fatalf("next start boundary %d, want 3", got)
	}
	if got := remapUTF8Boundary(raw, 2, true); got != 4 {
		t.Fatalf("final boundary %d, want 4", got)
	}
}
