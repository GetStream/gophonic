// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package arena

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestRegionsAndLifetime(t *testing.T) {
	a, err := New(1, 17, 0, 1025)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	lengths := []int{1, 17, 0, 1025}
	views := make([][]float32, len(lengths))
	for i, n := range lengths {
		views[i] = a.Take(n)
		if len(views[i]) != n || cap(views[i]) != n {
			t.Fatal("region can extend into its neighbor")
		}
		if n > 0 && uintptr(unsafe.Pointer(&views[i][0]))%64 != 0 {
			t.Fatal("unaligned region")
		}
		for j, x := range views[i] {
			if x != 0 {
				t.Fatal("mapping is not zeroed")
			}
			views[i][j] = float32(i*10000 + j)
		}
	}
	for range 3 {
		runtime.GC()
	}
	for i, v := range views {
		for j, x := range v {
			if x != float32(i*10000+j) {
				t.Fatal("overlapping or reclaimed region")
			}
		}
	}
	runtime.KeepAlive(a)
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if a.Bytes() != 0 {
		t.Fatal("close retained mapping")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSizesAndZero(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, sizes := range [][]int{{-1}, {maxInt}, {maxInt / 8, maxInt / 8, 64}} {
		if _, err := New(sizes...); err != ErrSize {
			t.Fatalf("New(%v): %v", sizes, err)
		}
	}
	a, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	if a.Bytes() != 0 || a.Take(0) != nil {
		t.Fatal("nonempty zero arena")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRejectsExhaustion(t *testing.T) {
	a, err := New(16)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	a.Take(16)
	defer func() {
		if recover() != ErrSize {
			t.Error("exhausted arena did not reject allocation")
		}
	}()
	a.Take(1)
}
