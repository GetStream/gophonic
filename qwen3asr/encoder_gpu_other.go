// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !(darwin && arm64)

package qwen3asr

import (
	"errors"

	"github.com/GetStream/gophonic/internal/safetensors"
)

type gpuEncoder struct{}

type gpuEncoderWorkspace struct{}

func loadGPUEncoder(*safetensors.Checkpoint, *encoder, string) (*gpuEncoder, error) {
	return nil, errors.New("qwen3asr: the GPU encoder needs darwin/arm64")
}

func (g *gpuEncoder) release() {}

func (g *gpuEncoder) newWorkspace() *gpuEncoderWorkspace { return &gpuEncoderWorkspace{} }

func (w *gpuEncoderWorkspace) close() {}

func (w *gpuEncoderWorkspace) features(int) ([]float32, error) { panic("unreachable") }

func (w *gpuEncoderWorkspace) encode(int) ([]float32, error) { panic("unreachable") }
