// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"reflect"
	"testing"
)

func TestModelCloseRetiresCodecAndMappedViews(t *testing.T) {
	unmapped := 0
	m := &Model{codec: &codec{firstTable: []float32{1}},
		textEmbed: []uint16{2}, codecEmbed: []uint16{3}, bosRow: []float32{4},
		unmap: []func() error{func() error { unmapped++; return nil }}}
	m.cpRows[0] = []float32{5}
	for range 2 {
		if err := m.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if unmapped != 1 || !reflect.DeepEqual(m, &Model{}) {
		t.Fatal("Close retained model storage or repeated unmapping")
	}
	if _, err := NewSynthesizer(m, LaneOptions{}); err == nil {
		t.Fatal("closed model opened a lane")
	}
	if _, err := NewSynthesizer(nil, LaneOptions{}); err == nil {
		t.Fatal("nil model opened a lane")
	}
	if err := (*Model)(nil).Close(); err != nil {
		t.Fatal(err)
	}
}
