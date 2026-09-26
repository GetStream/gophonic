// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import "testing"

func TestServerThreads(t *testing.T) {
	for _, tt := range []struct{ explicit, requests, procs, want int }{
		{0, 1, 16, 0}, {0, 4, 16, 4}, {0, 8, 16, 2}, {0, 8, 3, 1},
		{0, 3, 16, 5}, {0, 2, 256, 64}, {7, 8, 16, 7},
	} {
		if got := serverThreads(tt.explicit, tt.requests, tt.procs); got != tt.want {
			t.Errorf("%+v: got %d", tt, got)
		}
	}
}
