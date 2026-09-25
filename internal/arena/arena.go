// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package arena owns bounded, pointer-free inference scratch. On Unix its
// payload is anonymous mapped memory, outside the Go heap and GC pacing.
// Other platforms use the mmap package's pointer-free heap fallback.
package arena

import (
	"errors"
	"runtime"
	"unsafe"

	"github.com/GetStream/gophonic/internal/mmap"
)

// Arena is a single mapping divided into 64-byte-aligned float32 regions.
// It must not be copied, used concurrently, reset, or grown. An owning lane
// replaces it as a unit when its capacity grows and closes it after its
// workers stop. Only numeric data may be stored in its payload.
//
// Take's slices do not keep the Arena alive. Owners must retain it and call
// runtime.KeepAlive(owner) after their last access to any slice. Close makes
// every slice invalid. A cleanup is a backstop for abandoned owners, not a
// replacement for Close at a lane's lifetime boundary.
type Arena struct {
	data    []byte
	offset  int
	cleanup runtime.Cleanup
}

var ErrSize = errors.New("arena: invalid or overflowing region size")

// New reserves a single zero-filled mapping for the given float32 region
// lengths, with at most 63 bytes of padding per region. Allocation failure
// leaves no mapping behind; no partially initialized arena is published.
func New(lengths ...int) (*Arena, error) {
	return newArena(4, lengths)
}

// NewBytes is New with byte-sized regions, for packed numeric weights.
func NewBytes(lengths ...int) (*Arena, error) { return newArena(1, lengths) }

func newArena(width int, lengths []int) (*Arena, error) {
	total := 0
	const maxInt = int(^uint(0) >> 1)
	for _, n := range lengths {
		if n < 0 || n > (maxInt-63)/width {
			return nil, ErrSize
		}
		size := (n*width + 63) &^ 63
		if size > maxInt-total {
			return nil, ErrSize
		}
		total += size
	}
	a := &Arena{}
	if total == 0 {
		return a, nil
	}
	data, err := mmap.Anonymous(total)
	if err != nil {
		return nil, err
	}
	a.data = data
	a.cleanup = runtime.AddCleanup(a, func(data []byte) { _ = mmap.Unmap(data) }, data)
	return a, nil
}

// Take bumps to the next region. The caller must use the same lengths and
// order passed to New. Its capacity equals its length, preventing append
// from overwriting the next region. There are no per-region heap allocations.
func (a *Arena) Take(n int) []float32 {
	if n < 0 || n > int(^uint(0)>>1)/4 {
		panic(ErrSize)
	}
	b := a.TakeBytes(4 * n)
	if n == 0 {
		return nil
	}
	v := unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n)
	runtime.KeepAlive(a)
	return v
}

// TakeBytes takes one aligned region, bounding its capacity to its length.
func (a *Arena) TakeBytes(n int) []byte {
	if a == nil || n < 0 || n > len(a.data)-a.offset || n > int(^uint(0)>>1)-63 {
		panic(ErrSize)
	}
	if n == 0 {
		return nil
	}
	size := (n + 63) &^ 63
	if size > len(a.data)-a.offset {
		panic(ErrSize)
	}
	start := a.offset
	a.offset += size
	return a.data[start : start+n : start+n]
}

// Bytes reports reserved payload bytes, including region alignment padding.
func (a *Arena) Bytes() int {
	if a == nil {
		return 0
	}
	return len(a.data)
}

// Close releases the complete mapping once. It must follow the last use of
// every view, including worker and assembly accesses.
func (a *Arena) Close() error {
	if a == nil || a.data == nil {
		return nil
	}
	if err := mmap.Unmap(a.data); err != nil {
		return err
	}
	a.cleanup.Stop()
	a.data = nil
	a.offset = 0
	return nil
}
