// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build unix

package wcache

import (
	"os"
	"syscall"
)

// lock takes an exclusive advisory lock on the file at path, waiting for
// its holder, in this process or another. Closing the file releases it, as
// does the holder's exit.
func lock(path string) *os.File {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil
	}
	for {
		err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX)
		if err != syscall.EINTR {
			break
		}
	}
	if err != nil {
		f.Close()
		return nil
	}
	return f
}
