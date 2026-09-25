// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !(darwin && arm64)

package qwen3lm

import (
	"errors"

	"github.com/GetStream/gophonic/internal/safetensors"
)

type gpuModel struct{}

type gpuWorkspace struct{}

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

func gpuSupports(*modelConfig) bool { return false }

func (w *gpuWorkspace) logitsInto(*Weights, []float32, []float32) error {
	return errors.New("qwen3: GPU backend unavailable")
}

func (g *gpuModel) hasHead() bool { return false }

func (g *gpuModel) maxPositions() int { return 0 }

// GPUAvailable reports whether a Metal GPU is present.
func GPUAvailable() bool { return false }
