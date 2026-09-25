// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build unix

package mmap

import (
	"os"
	"syscall"
	"unsafe"
)

func mapFile(f *os.File, off int64, n int, writable bool) ([]byte, error) {
	prot := syscall.PROT_READ
	if writable {
		prot |= syscall.PROT_WRITE
	}
	return syscall.Mmap(int(f.Fd()), off, n, prot, syscall.MAP_SHARED)
}

func mapAnonymous(n int) ([]byte, error) {
	return syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
}

func unmap(b []byte) error { return syscall.Munmap(b) }

func flush(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	_, _, errno := syscall.Syscall(syscall.SYS_MSYNC, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), syscall.MS_ASYNC)
	if errno != 0 {
		return errno
	}
	return nil
}
