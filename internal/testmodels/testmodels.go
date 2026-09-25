// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package testmodels finds the model files that model-level tests and
// benchmarks use. Every model lives in one directory: models/ at the
// repository root, or the directory named by GOPHONIC_MODELS. A test asks for
// a model by its file name and is skipped, with instructions, when the file is
// absent, so `go test ./...` runs every check whose model is present.
package testmodels

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// File names in the models directory. models/README.md lists where each one
// comes from; tools/fetch-models.sh downloads and converts them.
const (
	WhisperTinyEN  = "tiny.en.gophonic"
	WhisperBaseEN  = "base.en.gophonic"
	WhisperSmallEN = "small.en.gophonic"
	SmartTurn      = "smart-turn-v3.2.gophonic"
	TinyMel        = "tinymel.gophonic"
	// Qwen3 is the official Qwen/Qwen3-8B snapshot directory.
	Qwen3 = "Qwen3-8B"
	// Qwen3HelloReference is the BF16 PyTorch hidden state of "hello" that
	// qwen3/tools/reference_hidden.py writes.
	Qwen3HelloReference = "qwen3-8b-hello-reference.f32"
	// CLMHead is the converted CLM v0.1 head for Qwen3-8B.
	CLMHead = "CLM_v0.1-8B.gclm"
)

// Dir returns the models directory: $GOPHONIC_MODELS when set, otherwise
// models/ at the root of the source tree this package was built from.
func Dir() string {
	if dir := os.Getenv("GOPHONIC_MODELS"); dir != "" {
		return dir
	}
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "models")
}

// Path returns the path of model name in Dir, skipping tb when it is absent.
func Path(tb testing.TB, name string) string {
	tb.Helper()
	path := filepath.Join(Dir(), name)
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("model %s is not in %s; run tools/fetch-models.sh or see models/README.md", name, Dir())
	}
	return path
}
