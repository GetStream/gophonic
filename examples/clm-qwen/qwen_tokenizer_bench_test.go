// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"os"
	"strings"
	"testing"
)

var qwenTokenizerBenchIDs []int

func BenchmarkQwenTokenizerEncodeInto(b *testing.B) {
	dir := os.Getenv("GOPHONIC_QWEN3_TOKENIZER")
	if dir == "" {
		b.Skip("set GOPHONIC_QWEN3_TOKENIZER to a local Qwen3 tokenizer directory")
	}
	tk, err := LoadQwenTokenizer(dir)
	if err != nil {
		b.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		text string
	}{
		{"short", "hello"},
		{"turn", "The Moon's gravitational pull explains Earth's tides."},
		{"long", strings.Repeat("Qwen3 state/action ranking 2026!\r\n", 128)},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.Run("gophonic_into", func(b *testing.B) {
				dst := make([]int, 0, len(tc.text)*4+32)
				var ws TokenizerWorkspace
				if _, err := tk.EncodeInto(tc.text, dst, &ws); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.SetBytes(int64(len(tc.text)))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var err error
					dst, err = tk.EncodeInto(tc.text, dst[:0], &ws)
					if err != nil {
						b.Fatal(err)
					}
				}
				qwenTokenizerBenchIDs = dst
			})
		})
	}
}
