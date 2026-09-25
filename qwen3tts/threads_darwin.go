// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3tts

import "syscall"

// matrixUnits reports the streaming matrix units: Apple silicon has one per
// performance-core cluster. Zero means unknown.
func matrixUnits() int {
	cores, err := syscall.SysctlUint32("hw.perflevel0.physicalcpu")
	if err != nil || cores == 0 {
		return 0
	}
	perCluster, err := syscall.SysctlUint32("hw.perflevel0.cpusperl2")
	if err != nil || perCluster == 0 {
		return 0
	}
	return int((cores + perCluster - 1) / perCluster)
}
