// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin

package qwen3lm

import "syscall"

// PerformanceCores reports Apple silicon's performance-core count. Efficiency
// cores share a slower SME unit and would straggle behind balanced shards.
func PerformanceCores() int {
	n, err := syscall.SysctlUint32("hw.perflevel0.physicalcpu")
	if err != nil || n == 0 {
		return 0
	}
	return int(n)
}
