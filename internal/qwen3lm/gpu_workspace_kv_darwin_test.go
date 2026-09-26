// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"os"
	"testing"

	"github.com/GetStream/gophonic/internal/qwen3lm/lmtest"
	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/thesyncim/vibejson"
)

// TestOfficialASRWorkspaceKVIsLazy checks that private-prefix continuation
// uses only the lane-owned cache, while fresh and shared-prefix passes lazily
// allocate a workspace cache and preserve the corresponding hidden states.
func TestOfficialASRWorkspaceKVIsLazy(t *testing.T) {
	path := testmodels.Path(t, testmodels.Qwen3ASR)
	raw, err := os.ReadFile(path + "/config.json")
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Thinker struct {
			Text TextConfig `json:"text_config"`
		} `json:"thinker_config"`
	}
	if err := vibejson.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	// Qwen3-ASR's default MRoPE axes coincide for this text-only decoder.
	cfg.Thinker.Text.RopeScaling = nil
	m, err := Load(path, LoadOptions{Format: WeightsGPUQ8, Prefix: "thinker.model.", Config: &cfg.Thinker.Text})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.Release)
	eval, err := NewEvaluator(m)
	if err != nil {
		t.Fatal(err)
	}

	newWorkspace := func() *Workspace {
		t.Helper()
		ws, err := eval.NewWorkspace(1)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := ws.Close(); err != nil {
				t.Errorf("close workspace: %v", err)
			}
		})
		if ws.gpu.kc != nil || ws.gpu.vc != nil {
			t.Fatal("new workspace eagerly allocated its K/V cache")
		}
		return ws
	}

	const capacity = 8
	ids := []int{100, 101, 102}
	width := m.cfg.hidden
	prefix, err := eval.NewPrefixKV(capacity)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = prefix.Close() })
	privateWS := newWorkspace()
	private := make([]float32, width)
	if err := eval.HiddenLastExtendInto(prefix, 0, ids[:2], private, privateWS); err != nil {
		t.Fatal(err)
	}
	if privateWS.gpu.kc != nil || privateWS.gpu.vc != nil {
		t.Fatal("private-prefix continuation allocated a workspace K/V cache")
	}
	privateLane, err := eval.NewPrefixKV(capacity)
	if err != nil {
		t.Fatal(err)
	}
	privateLane.CopyPrefix(prefix, len(ids)-1)
	if err := eval.HiddenLastExtendInto(privateLane, len(ids)-1, ids[2:], private, privateWS); err != nil {
		_ = privateLane.Close()
		t.Fatal(err)
	}
	_ = privateLane.Close()

	fresh := make([]float32, width)
	if err := eval.HiddenLastInto(ids, fresh, privateWS); err != nil {
		t.Fatal(err)
	}
	if privateWS.gpu.kc == nil || privateWS.gpu.vc == nil {
		t.Fatal("fresh pass did not allocate its workspace K/V cache")
	}
	workspaceKC, workspaceVC := privateWS.gpu.kc, privateWS.gpu.vc
	if want := 4 * m.cfg.layers * gpuPositions * m.cfg.kvDim; len(privateWS.gpu.kc.Bytes()) != want || len(privateWS.gpu.vc.Bytes()) != want {
		t.Fatalf("workspace K/V cache sizes = %d/%d, want %d bytes each", len(privateWS.gpu.kc.Bytes()), len(privateWS.gpu.vc.Bytes()), want)
	}
	if cos, _ := lmtest.VectorParity(fresh, private); !(cos >= 0.99999) {
		t.Fatalf("fresh vs private-prefix hidden cosine %.7f", cos)
	}

	privateAgain, err := eval.NewPrefixKV(capacity)
	if err != nil {
		t.Fatal(err)
	}
	privateAgain.CopyPrefix(prefix, len(ids)-1)
	wantPrivateAgain := make([]float32, width)
	if err := eval.HiddenLastExtendInto(privateAgain, len(ids)-1, ids[2:], wantPrivateAgain, privateWS); err != nil {
		_ = privateAgain.Close()
		t.Fatal(err)
	}
	_ = privateAgain.Close()
	if cos, _ := lmtest.VectorParity(fresh, wantPrivateAgain); !(cos >= 0.99999) {
		t.Fatalf("private→fresh→private workspace hidden cosine %.7f", cos)
	}

	sharedWS := newWorkspace()
	seqs := [][]int{{ids[2]}, {ids[2] + 1}}
	shared := [][]float32{make([]float32, width), make([]float32, width)}
	if err := eval.HiddenLastSharedInto(prefix, seqs, shared, sharedWS); err != nil {
		t.Fatal(err)
	}
	if sharedWS.gpu.kc == nil || sharedWS.gpu.vc == nil {
		t.Fatal("shared-prefix batch did not allocate its workspace K/V cache")
	}
	for lane, seq := range seqs {
		privateLane, err := eval.NewPrefixKV(capacity)
		if err != nil {
			t.Fatal(err)
		}
		privateLane.CopyPrefix(prefix, len(ids)-1)
		want := make([]float32, width)
		if err := eval.HiddenLastExtendInto(privateLane, len(ids)-1, seq, want, privateWS); err != nil {
			_ = privateLane.Close()
			t.Fatal(err)
		}
		_ = privateLane.Close()
		if cos, _ := lmtest.VectorParity(shared[lane], want); !(cos >= 0.99999) {
			t.Fatalf("shared vs private lane %d hidden cosine %.7f", lane, cos)
		}
	}
	if privateWS.gpu.kc != workspaceKC || privateWS.gpu.vc != workspaceVC {
		t.Fatal("private-prefix continuation replaced the workspace K/V buffers after reuse")
	}
	if err := sharedWS.Close(); err != nil {
		t.Fatal(err)
	}

	failedWS := newWorkspace()
	if err := eval.HiddenLastSharedInto(prefix, [][]int{{}}, [][]float32{make([]float32, width)}, failedWS); err == nil {
		t.Fatal("empty shared sequence unexpectedly succeeded")
	}
	if failedWS.gpu.kc != nil || failedWS.gpu.vc != nil {
		t.Fatal("invalid call allocated a workspace K/V cache before validation")
	}
}
