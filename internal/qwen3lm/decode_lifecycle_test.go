// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import "testing"

func TestDecoderCloseDropsEvaluator(t *testing.T) {
	d := &Decoder{e: &Evaluator{}, rows: 128, heads: 1, tables: 0, gpu: &gpuDecoder{}}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	if *d != (Decoder{}) {
		t.Fatal("Close retained decoder state", d)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("repeated Close: %v", err)
	}
}
