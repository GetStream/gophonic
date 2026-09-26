// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !(darwin && arm64)

package qwen3lm

import (
	"errors"

	"github.com/GetStream/gophonic/internal/safetensors"
)

type gpuModel struct{}

type gpuWorkspace struct {
	tail  []float32
	probe *Probe
}

type gpuPrefix struct{}

func (g *gpuModel) newPrefix(int) (*gpuPrefix, error) {
	return nil, errors.New("qwen3: the GPU backend requires darwin/arm64")
}

func (m *Weights) loadGPU(*safetensors.Checkpoint, int, string) error {
	return errors.New("qwen3: the GPU backend requires darwin/arm64")
}

func (g *gpuModel) newWorkspace() (*gpuWorkspace, error) {
	return nil, errors.New("qwen3: the GPU backend requires darwin/arm64")
}

func (w *gpuWorkspace) batch(*Weights, [][]int, [][]float32, *gpuPrefix, int, bool, Embeds) error {
	return errors.New("qwen3: the GPU backend requires darwin/arm64")
}

func (w *gpuWorkspace) release() {}

func (m *Weights) releaseGPU() {}

func (p *gpuPrefix) copyFrom(*gpuPrefix, int, int) {}

func (p *gpuPrefix) release() {}

func (p *gpuPrefix) recurrent() bool { return false }

func (p *gpuPrefix) copyStates(*gpuPrefix, int) {}

func (e *Evaluator) rewind(*PrefixKV, int, bool, *Workspace) error { return nil }

func (m *Weights) loadHybrid(*safetensors.Checkpoint) error {
	return errors.New("qwen3: Qwen3.5 models need the GPU backend (darwin/arm64)")
}

func gpuSupports(*modelConfig) bool { return false }

func (w *gpuWorkspace) logitsInto(*Weights, []float32, []float32) error {
	return errors.New("qwen3: GPU backend unavailable")
}

func (g *gpuModel) hasHead() bool { return false }

func (g *gpuModel) maxPositions() int { return 0 }

// GPUAvailable reports whether a Metal GPU is present.
func GPUAvailable() bool { return false }

func (w *gpuWorkspace) logitsRowsInto(*Weights, []float32, []float32, int) error {
	return errors.New("qwen3: GPU backend unavailable")
}

type gpuDecoder struct{}

func (g *gpuModel) newDecoder([][]float32, [][]float32, int, []float32) (*gpuDecoder, error) {
	return nil, errors.New("qwen3: no GPU")
}

func (d *gpuDecoder) release() {}

func (w *gpuWorkspace) decode(*Weights, *gpuDecoder, *gpuPrefix, int, []int, Embeds, Sampling, []int, []float32) error {
	return errors.New("qwen3: no GPU")
}
