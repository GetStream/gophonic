// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

// decoderPrefix records provenance only: the numeric audio KV remains in
// the lane's existing bounded PrefixKV arena. Reuse requires the immediately
// preceding encode, the same static prompt and arithmetic anchor, and the
// encoder's bit-exact validation of the embedding prefix.
type decoderPrefix struct {
	anchor, rows int
	revision     uint64
	valid        bool
}

func (p *decoderPrefix) reset() { *p = decoderPrefix{} }

func (p *decoderPrefix) reusable(anchor, stableRows int, revision uint64) int {
	if !p.valid || p.anchor != anchor || p.revision+1 != revision {
		return 0
	}
	return min(p.rows, stableRows)
}

func (p *decoderPrefix) remember(anchor, rows int, revision uint64) {
	*p = decoderPrefix{anchor: anchor, rows: rows, revision: revision, valid: true}
}
