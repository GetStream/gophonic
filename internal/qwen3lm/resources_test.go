// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"reflect"
	"testing"

	"github.com/GetStream/gophonic/internal/q8gemm"
)

func TestPrefixKVCloseRetiresOwner(t *testing.T) {
	kv := &PrefixKV{
		owner: &Evaluator{}, tokens: []int{1, 2}, packs: []prefixPack{{}},
	}
	if err := kv.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(kv, &PrefixKV{}) {
		t.Fatal("Close retained prefix state")
	}
}

func TestWorkspaceCloseRetiresModelAndScratch(t *testing.T) {
	ws := &Workspace{
		owner:       &Evaluator{},
		prefix:      &PrefixKV{},
		embeds:      Embeds{Rows: []float32{1}},
		tail:        []float32{2},
		probe:       &Probe{Probs: []float32{3}},
		attnItems:   []attentionItem{{}},
		attnScratch: []attentionScratch{{}},
		tiles:       []*q8gemm.Workspace{{}},
		tilesI8:     []*q8gemm.WorkspaceI8{{}},
		rotated:     []float32{4},
		scratch:     []*q8gemm.Scratch{{}},
		oneSeq:      [1][]int{{1}},
		oneDst:      [1][]float32{{5}},
		rewound:     []float32{6},
	}
	ws.op.ws = ws
	if err := ws.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ws, &Workspace{}) {
		t.Fatal("Close retained workspace state")
	}
}

func TestWeightsReleaseRetiresPayloadAndRejectsLazyHead(t *testing.T) {
	m := &Weights{
		cfg:       modelConfig{hidden: 4096, invFreq: []float64{1}},
		format:    WeightsF16,
		prefix:    "model.",
		dir:       "released-model",
		headName:  "lm_head.weight",
		embed:     []uint16{1},
		finalNorm: []float32{2},
		layers:    []modelLayer{{dnDT: []float32{3}}},
		head:      linear{rot: &rotation{block: 64}},
	}
	m.Release()
	if m.memory != nil || m.headMemory != nil || m.embed != nil || m.finalNorm != nil || m.layers != nil || m.head != (linear{}) {
		t.Fatal("Release retained weights payload")
	}
	if m.headName != "" || m.dir != "" || !m.released || m.cfg.hidden != 4096 || m.format != WeightsF16 {
		t.Fatal("Release did not retire lazy-head state or preserve diagnostics")
	}
	if m.HasHead() {
		t.Fatal("released weights report a head")
	}
	if err := m.LoadHead("lm_head.weight"); err == nil {
		t.Fatal("LoadHead succeeded after Release")
	}
	m.Release()
}
