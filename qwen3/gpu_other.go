// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !(darwin && arm64)

package qwen3

import "errors"

type gpuModel struct{}

type gpuWorkspace struct{}

type gpuPrefix struct{}

func (g *gpuModel) newPrefix(int) (*gpuPrefix, error) {
	return nil, errors.New("qwen3: the GPU backend requires darwin/arm64")
}

func (m *Weights) loadGPU(*safetensors, int) error {
	return errors.New("qwen3: the GPU backend requires darwin/arm64")
}

func (g *gpuModel) newWorkspace() (*gpuWorkspace, error) {
	return nil, errors.New("qwen3: the GPU backend requires darwin/arm64")
}

func (w *gpuWorkspace) batch(*Weights, [][]int, [][]float32, *gpuPrefix, int, bool) error {
	return errors.New("qwen3: the GPU backend requires darwin/arm64")
}

func (w *gpuWorkspace) release() {}

func (m *Weights) releaseGPU() {}

func (p *gpuPrefix) copyFrom(*gpuPrefix, int, int) {}

func (p *gpuPrefix) release() {}
