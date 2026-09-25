// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !unix

package mmap

import (
	"errors"
	"os"
)

func mapFile(*os.File, int64, int, bool) ([]byte, error) {
	return nil, errors.New("mmap: unsupported platform")
}

func mapAnonymous(n int) ([]byte, error) { return make([]byte, n), nil }

func unmap([]byte) error { return nil }

func flush([]byte) error { return nil }
