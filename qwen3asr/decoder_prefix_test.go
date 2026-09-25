// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import "testing"

func TestDecoderPrefixProvenance(t *testing.T) {
	var p decoderPrefix
	if n := p.reusable(17, 208, 1); n != 0 {
		t.Fatal("uninitialized decoder cache reused")
	}
	p.remember(17, 210, 7)
	for _, tc := range []struct {
		anchor, stable int
		revision       uint64
		want           int
	}{
		{17, 208, 8, 208}, {17, 416, 8, 210},
		{17, 0, 8, 0}, {16, 208, 8, 0}, {18, 208, 8, 0},
		{17, 208, 7, 0}, {17, 208, 9, 0},
	} {
		if n := p.reusable(tc.anchor, tc.stable, tc.revision); n != tc.want {
			t.Fatalf("%+v: reused %d", tc, n)
		}
	}
	// An encode without a completed decoder pass advances the revision,
	// rejecting KV that predates the encoder's immediately previous input.
	if n := p.reusable(17, 208, 10); n != 0 {
		t.Fatal("reused decoder rows after an abandoned encode")
	}
	p.reset()
	if n := p.reusable(17, 208, 8); n != 0 {
		t.Fatal("reset retained reusable rows")
	}
}
