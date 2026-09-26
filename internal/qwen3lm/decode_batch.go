// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// DecodeBatchWorkspace owns temporary scratch for DecodeBatchInto. It is
// private to one caller and must not be used concurrently. PrefixKV values
// passed to DecodeBatchInto remain private to their respective lanes; callers
// transfer exclusive ownership of them for the duration of the call.
type DecodeBatchWorkspace struct {
	owner    *Evaluator
	capacity int
	gpu      *gpuDecodeBatchWorkspace
}

// NewDecodeBatchWorkspace creates scratch for up to capacity independent
// next-token decodes. The current batched decoder requires Q8B Metal weights
// with a language-model head; scalar evaluation remains available for every
// other format.
func (e *Evaluator) NewDecodeBatchWorkspace(capacity int) (*DecodeBatchWorkspace, error) {
	if e == nil || e.m == nil {
		return nil, errors.New("qwen3: nil evaluator")
	}
	if capacity < 1 || capacity > 8 {
		return nil, fmt.Errorf("qwen3: decode batch capacity %d outside [1,8]", capacity)
	}
	gpu, err := e.m.newDecodeBatchWorkspace(capacity)
	if err != nil {
		return nil, err
	}
	return &DecodeBatchWorkspace{owner: e, capacity: capacity, gpu: gpu}, nil
}

// DecodeBatchInto advances each independent prefix by exactly one token,
// writing each lane's post-final-RMSNorm hidden state and logits. Every lane
// retains its own PrefixKV. All inputs, prefix ownership, capacities, and
// destination shapes are validated before GPU work begins or any prefix is
// advanced. The caller must exclusively own each prefix and this workspace
// until the call returns.
func (e *Evaluator) DecodeBatchInto(kvs []*PrefixKV, tokens []int, hidden, logits [][]float32, ws *DecodeBatchWorkspace) error {
	defer runtime.KeepAlive(ws)
	if e == nil || e.m == nil {
		return errors.New("qwen3: nil evaluator")
	}
	if ws == nil || ws.owner != e || ws.gpu == nil {
		return errors.New("qwen3: nil, closed, or foreign decode batch workspace")
	}
	n := len(kvs)
	if n == 0 || n != len(tokens) || n != len(hidden) || n != len(logits) {
		return fmt.Errorf("qwen3: decode batch has %d prefixes, %d tokens, %d hidden rows, and %d logits rows", n, len(tokens), len(hidden), len(logits))
	}
	if n > ws.capacity {
		return fmt.Errorf("qwen3: decode batch has %d lanes, workspace capacity is %d", n, ws.capacity)
	}
	c := &e.m.cfg
	for lane, kv := range kvs {
		if kv == nil || kv.owner != e || kv.gpu == nil || kv.capacity < 1 {
			return fmt.Errorf("qwen3: decode prefix %d is nil, closed, or belongs to another evaluator", lane)
		}
		for prev := 0; prev < lane; prev++ {
			if kv == kvs[prev] {
				return fmt.Errorf("qwen3: decode prefix %d is duplicated", lane)
			}
		}
		if len(kv.tokens) >= kv.capacity || len(kv.tokens) >= e.m.gpu.maxPositions() {
			return fmt.Errorf("qwen3: decode prefix %d has no room for another token", lane)
		}
		if tokens[lane] < 0 || tokens[lane] >= c.vocab {
			return fmt.Errorf("qwen3: token %d in decode lane %d is outside vocabulary", tokens[lane], lane)
		}
		if len(hidden[lane]) != c.hidden || len(logits[lane]) != c.vocab {
			return fmt.Errorf("qwen3: decode lane %d has hidden/logits widths %d/%d, want %d/%d", lane, len(hidden[lane]), len(logits[lane]), c.hidden, c.vocab)
		}
		if decodeFloatRowsOverlap(hidden[lane], logits[lane]) {
			return errors.New("qwen3: decode batch destinations overlap")
		}
		for prev := 0; prev < lane; prev++ {
			if decodeFloatRowsOverlap(hidden[lane], hidden[prev]) || decodeFloatRowsOverlap(hidden[lane], logits[prev]) ||
				decodeFloatRowsOverlap(logits[lane], hidden[prev]) || decodeFloatRowsOverlap(logits[lane], logits[prev]) {
				return errors.New("qwen3: decode batch destinations overlap")
			}
		}
	}
	if err := ws.gpu.decode(e.m, kvs, tokens, hidden, logits); err != nil {
		return err
	}
	for lane, kv := range kvs {
		kv.tokens = append(kv.tokens, tokens[lane])
	}
	return nil
}

// Close releases scratch owned by ws. Repeated calls are harmless.
func (ws *DecodeBatchWorkspace) Close() error {
	if ws == nil || ws.gpu == nil {
		return nil
	}
	ws.gpu.release()
	ws.gpu = nil
	ws.capacity = 0
	return nil
}

func decodeFloatRowsOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	a0 := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	b0 := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	a1 := a0 + uintptr(len(a))*unsafe.Sizeof(float32(0))
	b1 := b0 + uintptr(len(b))*unsafe.Sizeof(float32(0))
	return a0 < b1 && b0 < a1
}
