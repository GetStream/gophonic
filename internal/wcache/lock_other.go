// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build !unix

package wcache

import "os"

func lock(string) *os.File { return nil }
