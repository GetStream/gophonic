// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package mmap maps files and anonymous memory. Mappings start on a page
// boundary, so the Metal backend can make GPU buffers of them without a
// copy.
package mmap

import "os"

// File maps n bytes of f from offset off, which must be a multiple of the
// page size. A writable mapping is shared: stores reach the file.
func File(f *os.File, off int64, n int, writable bool) ([]byte, error) {
	return mapFile(f, off, n, writable)
}

// Anonymous maps n zeroed bytes of private memory.
func Anonymous(n int) ([]byte, error) { return mapAnonymous(n) }

// Flush starts writing the modified pages of a shared file mapping to the
// file, without waiting for the writes.
func Flush(b []byte) error { return flush(b) }

// Unmap releases a mapping made by File or Anonymous.
func Unmap(b []byte) error { return unmap(b) }

// PageSize is the system page size.
var PageSize = os.Getpagesize()
