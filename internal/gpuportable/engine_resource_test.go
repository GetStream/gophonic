// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gpuportable

import (
	"context"
	"errors"
	"testing"

	"github.com/gogpu/wgpu"
)

type fakeReadbackMapping struct {
	mapErr     error
	rangeErr   error
	unmapErr   error
	unmapCalls int
	rangeCalls int
}

func (m *fakeReadbackMapping) Map(context.Context, wgpu.MapMode, uint64, uint64) error {
	return m.mapErr
}

func (m *fakeReadbackMapping) MappedRange(uint64, uint64) (*wgpu.MappedRange, error) {
	m.rangeCalls++
	return nil, m.rangeErr
}

func (m *fakeReadbackMapping) Unmap() error {
	m.unmapCalls++
	return m.unmapErr
}

func TestCopyMappedReadbackCancelsFailedMap(t *testing.T) {
	mapErr := errors.New("map timed out")
	unmapErr := errors.New("cancel map failed")
	mapping := &fakeReadbackMapping{mapErr: mapErr, unmapErr: unmapErr}

	err := copyMappedReadback(context.Background(), mapping, 16, make([]byte, 16))
	if !errors.Is(err, mapErr) || !errors.Is(err, unmapErr) {
		t.Fatalf("copyMappedReadback error = %v, want both map and unmap errors", err)
	}
	if mapping.unmapCalls != 1 {
		t.Fatalf("Unmap calls = %d, want 1", mapping.unmapCalls)
	}
	if mapping.rangeCalls != 0 {
		t.Fatalf("MappedRange calls = %d, want 0 after failed map", mapping.rangeCalls)
	}
}

func TestCopyMappedReadbackUnmapsAfterRangeFailure(t *testing.T) {
	rangeErr := errors.New("mapped range failed")
	mapping := &fakeReadbackMapping{rangeErr: rangeErr}

	err := copyMappedReadback(context.Background(), mapping, 16, make([]byte, 16))
	if !errors.Is(err, rangeErr) {
		t.Fatalf("copyMappedReadback error = %v, want %v", err, rangeErr)
	}
	if mapping.unmapCalls != 1 {
		t.Fatalf("Unmap calls = %d, want 1", mapping.unmapCalls)
	}
}
