// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build rust

package gpuportable

import (
	"errors"
	"testing"
)

func TestRustTagFailsBeforeOpeningRuntime(t *testing.T) {
	if !errors.Is(runtimeBuildError(), ErrRustBindingBroken) {
		t.Fatal("rust build must reject the known-broken pinned binding before calling wgpu-native")
	}
}
