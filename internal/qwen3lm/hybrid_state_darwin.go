// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

// A hybrid model's recurrent state cannot be cut back to a prefix the way
// keys and values can: rewinding a conversation (a speculative message
// dropped, a reply cut short) needs the state as it was. A PrefixKV keeps
// snapshots at the points where an extension of more than one token began,
// which are where conversations branch, and rewinds to the latest snapshot
// at or before the point asked for, evaluating the tokens in between again.

// recurrent reports whether p holds a recurrent state.
func (p *gpuPrefix) recurrent() bool { return p.state != nil }

// maxSnapshots bounds a PrefixKV's snapshots (about 66 MB each for
// Qwen3.6-35B-A3B).
const maxSnapshots = 8

// rewind makes kv's state that of its first keep tokens, and, with mark,
// snapshots it there.
func (e *Evaluator) rewind(kv *PrefixKV, keep int, mark bool, ws *Workspace) error {
	g := kv.gpu
	if g.state.pos != keep {
		var from *gpuState
		for _, s := range g.snaps {
			if s.pos <= keep && (from == nil || s.pos > from.pos) {
				from = s
			}
		}
		if from != nil {
			g.state.copyFrom(from)
		} else {
			g.state.reset()
		}
		if g.state.pos < keep {
			// Evaluate the tokens between the snapshot and keep again. kv's
			// tokens are rewritten in place with the same values.
			if cap(ws.rewound) < e.m.cfg.hidden {
				ws.rewound = make([]float32, e.m.cfg.hidden)
			}
			if err := e.HiddenLastExtendInto(kv, g.state.pos, kv.tokens[g.state.pos:keep], ws.rewound, ws); err != nil {
				return err
			}
		}
	}
	// Snapshots past keep describe tokens about to be replaced.
	live := g.snaps[:0]
	for _, s := range g.snaps {
		if s.pos <= keep {
			live = append(live, s)
		} else {
			g.spare = append(g.spare, s)
		}
	}
	g.snaps = live
	if !mark {
		return nil
	}
	for _, s := range g.snaps {
		if s.pos == keep {
			return nil
		}
	}
	var s *gpuState
	switch {
	case len(g.spare) > 0:
		s, g.spare = g.spare[len(g.spare)-1], g.spare[:len(g.spare)-1]
	case len(g.snaps) < maxSnapshots:
		var err error
		if s, err = e.m.gpu.newState(); err != nil {
			return err
		}
	default:
		// The oldest snapshot but the first goes: the first is usually
		// where the conversation begins after its system prompt.
		oldest := 1
		for i := 2; i < len(g.snaps); i++ {
			if g.snaps[i].used < g.snaps[oldest].used {
				oldest = i
			}
		}
		s = g.snaps[oldest]
		g.snaps = append(g.snaps[:oldest], g.snaps[oldest+1:]...)
	}
	s.copyFrom(g.state)
	g.clock++
	s.used = g.clock
	g.snaps = append(g.snaps, s)
	return nil
}
