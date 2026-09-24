// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whispergemm

import "syscall"

func smeSupported() bool {
	v, err := syscall.SysctlUint32("hw.optional.arm.FEAT_SME")
	return err == nil && v == 1
}
