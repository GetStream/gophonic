// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package metal

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

// call6 and call9 call a C function through the runtime's libc trampoline
// on the system stack. Integer and pointer arguments only.
//
//go:linkname call6 syscall.syscall6X
func call6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

// rawCall6 skips the scheduler hand-off, for calls that never block.
//
//go:linkname rawCall6 syscall.rawSyscall6
func rawCall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

//go:linkname call9 syscall.syscall9
func call9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err syscall.Errno)

//go:cgo_import_dynamic libc_dlopen dlopen "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlsym dlsym "/usr/lib/libSystem.B.dylib"

var libc_dlopen_trampoline_addr uintptr
var libc_dlsym_trampoline_addr uintptr
