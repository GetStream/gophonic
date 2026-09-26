// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package qwen3lm

import (
	"fmt"
	"math"
	"syscall"
	"testing"
	"time"
)

func TestOfficialGPUDecodeBatchMatchesIndependentTokens(t *testing.T) {
	m := loadOfficialASRQ8B(t, true)
	e, err := NewEvaluator(m)
	if err != nil {
		t.Fatal(err)
	}
	ws, err := e.NewDecodeBatchWorkspace(8)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ws.Close(); err != nil {
			t.Errorf("close decode batch workspace: %v", err)
		}
	})
	scalarWS, err := e.NewWorkspace(1)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = scalarWS.Close() })

	const capacity = 12
	seeds := [][]int{{}, {100, 101}, {202, 203, 204, 205}, {306}, {407, 408, 409}, {510, 511, 512, 513, 514}, {}, {714, 715}}
	base := make([]*PrefixKV, len(seeds))
	for lane, seed := range seeds {
		base[lane], err = e.NewPrefixKV(capacity)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = base[lane].Close() })
		if len(seed) != 0 {
			h := make([]float32, m.cfg.hidden)
			if err := e.HiddenLastExtendInto(base[lane], 0, seed, h, scalarWS); err != nil {
				t.Fatalf("seed lane %d: %v", lane, err)
			}
		}
	}

	for _, active := range []int{1, 2, 3, 4, 5, 8} {
		t.Run(fmt.Sprintf("active=%d", active), func(t *testing.T) {
			kvs := make([]*PrefixKV, active)
			refs := make([]*PrefixKV, active)
			t.Cleanup(func() {
				for lane := range active {
					_ = kvs[lane].Close()
					_ = refs[lane].Close()
				}
			})
			tokens := make([]int, active)
			hidden := make([][]float32, active)
			logits := make([][]float32, active)
			wantHidden := make([][]float32, active)
			wantLogits := make([][]float32, active)
			for lane := range active {
				kvs[lane], err = e.NewPrefixKV(capacity)
				if err != nil {
					t.Fatal(err)
				}
				refs[lane], err = e.NewPrefixKV(capacity)
				if err != nil {
					t.Fatal(err)
				}
				kvs[lane].CopyPrefix(base[lane], len(seeds[lane]))
				refs[lane].CopyPrefix(base[lane], len(seeds[lane]))
				tokens[lane] = 900 + active*8 + lane
				hidden[lane] = make([]float32, m.cfg.hidden)
				logits[lane] = make([]float32, m.cfg.vocab)
				wantHidden[lane] = make([]float32, m.cfg.hidden)
				wantLogits[lane] = make([]float32, m.cfg.vocab)
				if err := e.HiddenLastExtendInto(refs[lane], len(refs[lane].Tokens()), []int{tokens[lane]}, wantHidden[lane], scalarWS); err != nil {
					t.Fatalf("scalar lane %d: %v", lane, err)
				}
				if err := e.LogitsInto(wantHidden[lane], wantLogits[lane], scalarWS); err != nil {
					t.Fatalf("scalar logits lane %d: %v", lane, err)
				}
			}

			// Run both command-encoder modes against the same starting prefixes.
			// Concurrent compute encoders may reorder independent dispatches, so
			// this checks every explicit stage barrier against the serialized path.
			ws.gpu.concurrent = false
			if err := e.DecodeBatchInto(kvs, tokens, hidden, logits, ws); err != nil {
				t.Fatal(err)
			}
			serialHidden := make([][]float32, active)
			serialLogits := make([][]float32, active)
			for lane := range active {
				serialHidden[lane] = append([]float32(nil), hidden[lane]...)
				serialLogits[lane] = append([]float32(nil), logits[lane]...)
				kvs[lane].CopyPrefix(base[lane], len(seeds[lane]))
			}
			ws.gpu.concurrent = true
			if err := e.DecodeBatchInto(kvs, tokens, hidden, logits, ws); err != nil {
				t.Fatalf("concurrent decode: %v", err)
			}
			for lane := range active {
				if len(kvs[lane].Tokens()) != len(seeds[lane])+1 || kvs[lane].Tokens()[len(seeds[lane])] != tokens[lane] {
					t.Fatalf("lane %d prefix did not append its token: %v", lane, kvs[lane].Tokens())
				}
				maxAbs, rel, finite := gpuVectorError(serialHidden[lane], hidden[lane])
				if !finite || maxAbs > 1e-5 || rel > 1e-6 {
					t.Fatalf("lane %d serial/concurrent hidden mismatch: max_abs %.3g relative %.3g finite=%t", lane, maxAbs, rel, finite)
				}
				maxAbs, rel, finite = gpuVectorError(serialLogits[lane], logits[lane])
				if !finite || maxAbs > 1e-5 || rel > 1e-6 {
					t.Fatalf("lane %d serial/concurrent logits mismatch: max_abs %.3g relative %.3g finite=%t", lane, maxAbs, rel, finite)
				}
				maxAbs, rel, finite = gpuVectorError(wantHidden[lane], hidden[lane])
				if !finite || maxAbs > 1e-4 || rel > 1e-5 {
					t.Fatalf("lane %d hidden parity: max_abs %.3g relative %.3g finite=%t", lane, maxAbs, rel, finite)
				}
				maxAbs, rel, finite = gpuVectorError(wantLogits[lane], logits[lane])
				if !finite || maxAbs > 2e-4 || rel > 1e-5 {
					t.Fatalf("lane %d logits parity: max_abs %.3g relative %.3g finite=%t", lane, maxAbs, rel, finite)
				}

				// Continue both caches scalarly to detect bad per-lane K/V
				// offsets after the batched write.
				next := 1200 + lane
				gotNext, wantNext := make([]float32, m.cfg.hidden), make([]float32, m.cfg.hidden)
				if err := e.HiddenLastExtendInto(kvs[lane], len(kvs[lane].Tokens()), []int{next}, gotNext, scalarWS); err != nil {
					t.Fatalf("scalar continuation of batch lane %d: %v", lane, err)
				}
				if err := e.HiddenLastExtendInto(refs[lane], len(refs[lane].Tokens()), []int{next}, wantNext, scalarWS); err != nil {
					t.Fatalf("scalar reference continuation of lane %d: %v", lane, err)
				}
				maxAbs, rel, finite = gpuVectorError(wantNext, gotNext)
				if !finite || maxAbs > 1e-4 || rel > 1e-5 {
					t.Fatalf("lane %d continued hidden parity: max_abs %.3g relative %.3g finite=%t", lane, maxAbs, rel, finite)
				}
			}
		})
	}
}

func TestOfficialGPUDecodeBatchGreedyMatchesFullLogits(t *testing.T) {
	m := loadOfficialASRQ8B(t, true)
	e, err := NewEvaluator(m)
	if err != nil {
		t.Fatal(err)
	}
	ws, err := e.NewDecodeBatchWorkspace(8)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	scalarWS, err := e.NewWorkspace(1)
	if err != nil {
		t.Fatal(err)
	}
	defer scalarWS.Close()

	const capacity = 12
	seeds := [][]int{{}, {100, 101}, {202, 203, 204, 205}, {306}, {407, 408, 409}, {510, 511, 512, 513, 514}, {}, {714, 715}}
	base := make([]*PrefixKV, len(seeds))
	for lane, seed := range seeds {
		base[lane], err = e.NewPrefixKV(capacity)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = base[lane].Close() })
		if len(seed) != 0 {
			if err := e.HiddenLastExtendInto(base[lane], 0, seed, make([]float32, m.cfg.hidden), scalarWS); err != nil {
				t.Fatalf("seed lane %d: %v", lane, err)
			}
		}
	}

	for _, active := range []int{1, 2, 3, 4, 5, 8} {
		t.Run(fmt.Sprintf("active=%d", active), func(t *testing.T) {
			ref := make([]*PrefixKV, active)
			for lane := range active {
				ref[lane], err = e.NewPrefixKV(capacity)
				if err != nil {
					t.Fatal(err)
				}
				ref[lane].CopyPrefix(base[lane], len(seeds[lane]))
				t.Cleanup(func() { _ = ref[lane].Close() })
			}
			tokens := make([]int, active)
			hiddenRef := make([][]float32, active)
			logitsRef := make([][]float32, active)
			for lane := range active {
				tokens[lane] = 900 + active*8 + lane
				hiddenRef[lane] = make([]float32, m.cfg.hidden)
				logitsRef[lane] = make([]float32, m.cfg.vocab)
			}
			if err := e.DecodeBatchInto(ref, tokens, hiddenRef, logitsRef, ws); err != nil {
				t.Fatalf("full-logit reference: %v", err)
			}

			for _, gpuReduce := range []bool{false, true} {
				name := "mapped-cpu-argmax"
				if gpuReduce {
					name = "gpu-argmax"
				}
				t.Run(name, func(t *testing.T) {
					gotKV := make([]*PrefixKV, active)
					gotHidden := make([][]float32, active)
					next := make([]int, active)
					for lane := range active {
						gotKV[lane], err = e.NewPrefixKV(capacity)
						if err != nil {
							t.Fatal(err)
						}
						gotKV[lane].CopyPrefix(base[lane], len(seeds[lane]))
						t.Cleanup(func() { _ = gotKV[lane].Close() })
						gotHidden[lane] = make([]float32, m.cfg.hidden)
					}
					ws.gpu.argmaxGPU = gpuReduce
					if err := e.DecodeBatchGreedyInto(gotKV, tokens, gotHidden, next, ws); err != nil {
						t.Fatal(err)
					}
					for lane := range active {
						if next[lane] != greedyArgmax(logitsRef[lane]) {
							t.Errorf("lane %d next token %d, full-logit argmax %d", lane, next[lane], greedyArgmax(logitsRef[lane]))
						}
						if len(gotKV[lane].Tokens()) != len(seeds[lane])+1 || gotKV[lane].Tokens()[len(seeds[lane])] != tokens[lane] {
							t.Errorf("lane %d did not append input token: %v", lane, gotKV[lane].Tokens())
						}
						for i := range gotHidden[lane] {
							if math.Float32bits(gotHidden[lane][i]) != math.Float32bits(hiddenRef[lane][i]) {
								t.Fatalf("lane %d hidden value %d changed bits: %08x != %08x", lane, i, math.Float32bits(gotHidden[lane][i]), math.Float32bits(hiddenRef[lane][i]))
							}
						}
					}
				})
			}
		})
	}
}

func TestGPUDecodeBatchRejectsInvalidInputsBeforeMutation(t *testing.T) {
	m := loadOfficialASRQ8B(t, true)
	e, err := NewEvaluator(m)
	if err != nil {
		t.Fatal(err)
	}
	ws, err := e.NewDecodeBatchWorkspace(2)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	other, err := NewEvaluator(m)
	if err != nil {
		t.Fatal(err)
	}
	otherWS, err := other.NewDecodeBatchWorkspace(2)
	if err != nil {
		t.Fatal(err)
	}
	defer otherWS.Close()
	first, err := e.NewPrefixKV(4)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := e.NewPrefixKV(4)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()

	hidden := [][]float32{make([]float32, m.cfg.hidden), make([]float32, m.cfg.hidden)}
	logits := [][]float32{make([]float32, m.cfg.vocab), make([]float32, m.cfg.vocab)}
	for lane := range hidden {
		for i := range hidden[lane] {
			hidden[lane][i] = float32(100 + lane)
		}
		for i := range logits[lane] {
			logits[lane][i] = float32(200 + lane)
		}
	}
	shared := make([]float32, m.cfg.vocab)
	for i := range shared {
		shared[i] = 333
	}
	crossLane := make([]float32, m.cfg.hidden)
	for i := range crossLane {
		crossLane[i] = 444
	}
	assertUnchanged := func() {
		t.Helper()
		for lane, kv := range []*PrefixKV{first, second} {
			if len(kv.Tokens()) != 0 {
				t.Fatalf("invalid call mutated lane %d prefix: %v", lane, kv.Tokens())
			}
			for _, v := range hidden[lane] {
				if v != float32(100+lane) {
					t.Fatalf("invalid call mutated lane %d hidden destination", lane)
				}
			}
			for _, v := range logits[lane] {
				if v != float32(200+lane) {
					t.Fatalf("invalid call mutated lane %d logits destination", lane)
				}
			}
		}
	}

	cases := []struct {
		name    string
		eval    *Evaluator
		ws      *DecodeBatchWorkspace
		kvs     []*PrefixKV
		tok     []int
		hid     [][]float32
		log     [][]float32
		aliased []float32
	}{
		{name: "invalid-token", eval: e, ws: ws, kvs: []*PrefixKV{first, second}, tok: []int{7, m.cfg.vocab}, hid: hidden, log: logits},
		{name: "duplicate-prefix", eval: e, ws: ws, kvs: []*PrefixKV{first, first}, tok: []int{7, 8}, hid: hidden, log: logits},
		{name: "foreign-prefix", eval: other, ws: otherWS, kvs: []*PrefixKV{first}, tok: []int{7}, hid: [][]float32{hidden[0]}, log: [][]float32{logits[0]}},
		{name: "bad-output-width", eval: e, ws: ws, kvs: []*PrefixKV{first}, tok: []int{7}, hid: [][]float32{hidden[0][:1]}, log: [][]float32{logits[0]}},
		{name: "overlapping-outputs", eval: e, ws: ws, kvs: []*PrefixKV{first}, tok: []int{7}, hid: [][]float32{shared[:m.cfg.hidden]}, log: [][]float32{shared}, aliased: shared},
		{name: "cross-lane-overlap", eval: e, ws: ws, kvs: []*PrefixKV{first, second}, tok: []int{7, 8}, hid: [][]float32{crossLane, crossLane}, log: logits, aliased: crossLane},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.eval.DecodeBatchInto(tc.kvs, tc.tok, tc.hid, tc.log, tc.ws); err == nil {
				t.Fatal("invalid input unexpectedly succeeded")
			}
			for _, value := range tc.aliased {
				want := float32(333)
				if tc.name == "cross-lane-overlap" {
					want = 444
				}
				if value != want {
					t.Fatalf("invalid call mutated aliased destination: got %g want %g", value, want)
				}
			}
			assertUnchanged()
		})
	}
	if err := e.DecodeBatchGreedyInto([]*PrefixKV{first, second}, []int{7, 8}, hidden, []int{91}, ws); err == nil {
		t.Fatal("wrong next-token output length unexpectedly succeeded")
	}
	assertUnchanged()
	inputAndOutput := []int{7}
	if err := e.DecodeBatchGreedyInto([]*PrefixKV{first}, inputAndOutput, [][]float32{hidden[0]}, inputAndOutput, ws); err == nil {
		t.Fatal("next-token output aliasing input tokens unexpectedly succeeded")
	}
	assertUnchanged()
	prefixSlot := first.tokens[:cap(first.tokens)][:1]
	prefixSlot[0] = 515
	if err := e.DecodeBatchGreedyInto([]*PrefixKV{first}, []int{7}, [][]float32{hidden[0]}, prefixSlot, ws); err == nil {
		t.Fatal("next-token output aliasing prefix capacity unexpectedly succeeded")
	}
	if prefixSlot[0] != 515 {
		t.Fatal("invalid greedy call mutated prefix capacity")
	}
	assertUnchanged()

	closed, err := e.NewPrefixKV(4)
	if err != nil {
		t.Fatal(err)
	}
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	if err := e.DecodeBatchInto([]*PrefixKV{closed}, []int{7}, [][]float32{hidden[0]}, [][]float32{logits[0]}, ws); err == nil {
		t.Fatal("closed prefix unexpectedly succeeded")
	}
	assertUnchanged()

	full, err := e.NewPrefixKV(1)
	if err != nil {
		t.Fatal(err)
	}
	full.tokens = append(full.tokens, 17)
	if err := e.DecodeBatchInto([]*PrefixKV{full}, []int{7}, [][]float32{hidden[0]}, [][]float32{logits[0]}, ws); err == nil {
		t.Fatal("full prefix unexpectedly succeeded")
	}
	if len(full.Tokens()) != 1 || full.Tokens()[0] != 17 {
		t.Fatalf("full prefix changed after rejection: %v", full.Tokens())
	}
	_ = full.Close()
	assertUnchanged()

	if err := ws.Close(); err != nil {
		t.Fatal(err)
	}
	if err := e.DecodeBatchInto([]*PrefixKV{first}, []int{7}, [][]float32{hidden[0]}, [][]float32{logits[0]}, ws); err == nil {
		t.Fatal("closed workspace unexpectedly succeeded")
	}
	if err := e.DecodeBatchGreedyInto([]*PrefixKV{first}, []int{7}, [][]float32{hidden[0]}, []int{0}, ws); err == nil {
		t.Fatal("closed workspace greedy call unexpectedly succeeded")
	}
	assertUnchanged()
}

// BenchmarkGPUDecodeBatchStepPaired compares full 8-lane decoder steps while
// alternating serialized/concurrent attention scheduling and M8 versus two
// M4 projection tiles. Prefix lengths and private KV buffers stay fixed, so
// each iteration repeats the same next-token decode without copying caches.
func BenchmarkGPUDecodeBatchStepPaired(b *testing.B) {
	m := loadOfficialASRQ8B(b, true)
	e, err := NewEvaluator(m)
	if err != nil {
		b.Fatal(err)
	}
	ws, err := e.NewDecodeBatchWorkspace(8)
	if err != nil {
		b.Fatal(err)
	}
	defer ws.Close()
	scalarWS, err := e.NewWorkspace(1)
	if err != nil {
		b.Fatal(err)
	}
	defer scalarWS.Close()

	const lanes, prefixLen = 8, 32
	kvs := make([]*PrefixKV, lanes)
	tokens := make([]int, lanes)
	hidden := make([][]float32, lanes)
	logits := make([][]float32, lanes)
	for lane := range lanes {
		kvs[lane], err = e.NewPrefixKV(prefixLen + 1)
		if err != nil {
			b.Fatal(err)
		}
		defer kvs[lane].Close()
		seed := make([]int, prefixLen)
		for i := range seed {
			seed[i] = 100 + lane*prefixLen + i
		}
		if err := e.HiddenLastExtendInto(kvs[lane], 0, seed, make([]float32, m.cfg.hidden), scalarWS); err != nil {
			b.Fatalf("seed lane %d: %v", lane, err)
		}
		tokens[lane] = 500 + lane
		hidden[lane] = make([]float32, m.cfg.hidden)
		logits[lane] = make([]float32, m.cfg.vocab)
	}

	variants := []struct {
		name       string
		concurrent bool
		tileWidth  int
	}{
		{"m8-serial", false, 8},
		{"m8-concurrent", true, 8},
		{"m4x2-serial", false, 4},
		{"m4x2-concurrent", true, 4},
	}
	wantHidden := make([]float32, m.cfg.hidden*lanes)
	wantLogits := make([]float32, m.cfg.vocab*lanes)
	for variantIndex, variant := range variants {
		if err := ws.gpu.decodeModeTile(m, kvs, tokens, hidden, logits, variant.concurrent, variant.tileWidth); err != nil {
			b.Fatalf("%s parity warmup: %v", variant.name, err)
		}
		for lane := range lanes {
			if variantIndex == 0 {
				copy(wantHidden[lane*m.cfg.hidden:(lane+1)*m.cfg.hidden], hidden[lane])
				copy(wantLogits[lane*m.cfg.vocab:(lane+1)*m.cfg.vocab], logits[lane])
				continue
			}
			maxAbs, rel, finite := gpuVectorError(wantHidden[lane*m.cfg.hidden:(lane+1)*m.cfg.hidden], hidden[lane])
			if !finite || maxAbs > 1e-5 || rel > 1e-6 {
				b.Fatalf("%s lane %d hidden parity: max_abs %.3g relative %.3g finite=%t", variant.name, lane, maxAbs, rel, finite)
			}
			maxAbs, rel, finite = gpuVectorError(wantLogits[lane*m.cfg.vocab:(lane+1)*m.cfg.vocab], logits[lane])
			if !finite || maxAbs > 1e-5 || rel > 1e-6 {
				b.Fatalf("%s lane %d logits parity: max_abs %.3g relative %.3g finite=%t", variant.name, lane, maxAbs, rel, finite)
			}
		}
	}

	elapsed := make([]time.Duration, len(variants))
	b.ReportAllocs()
	b.ResetTimer()
	iteration := 0
	for b.Loop() {
		for step := range variants {
			index := step
			if iteration%2 != 0 {
				index = len(variants) - 1 - step
			}
			variant := variants[index]
			start := time.Now()
			err := ws.gpu.decodeModeTile(m, kvs, tokens, hidden, logits, variant.concurrent, variant.tileWidth)
			elapsed[index] += time.Since(start)
			if err != nil {
				b.Fatalf("%s decode: %v", variant.name, err)
			}
		}
		iteration++
	}
	b.StopTimer()
	for i, variant := range variants {
		b.ReportMetric(float64(elapsed[i].Nanoseconds())/float64(b.N)/1e6, variant.name+"-ms/step")
	}
}

// BenchmarkGPUDecodeBatchGreedyPaired compares the existing full-logit copy
// and host scan with mapped-logit scanning and a GPU stable argmax reduction.
// Each variant decodes the same eight private prefixes without advancing
// them; execution order alternates each iteration to limit drift bias.
func BenchmarkGPUDecodeBatchGreedyPaired(b *testing.B) {
	m := loadOfficialASRQ8B(b, true)
	e, err := NewEvaluator(m)
	if err != nil {
		b.Fatal(err)
	}
	ws, err := e.NewDecodeBatchWorkspace(8)
	if err != nil {
		b.Fatal(err)
	}
	defer ws.Close()
	scalarWS, err := e.NewWorkspace(1)
	if err != nil {
		b.Fatal(err)
	}
	defer scalarWS.Close()

	const lanes, prefixLen = 8, 32
	kvs := make([]*PrefixKV, lanes)
	tokens := make([]int, lanes)
	hidden := make([][]float32, lanes)
	logits := make([][]float32, lanes)
	gotHidden := make([][]float32, lanes)
	next := make([]int, lanes)
	for lane := range lanes {
		kvs[lane], err = e.NewPrefixKV(prefixLen + 1)
		if err != nil {
			b.Fatal(err)
		}
		defer kvs[lane].Close()
		seed := make([]int, prefixLen)
		for i := range seed {
			seed[i] = 100 + lane*prefixLen + i
		}
		if err := e.HiddenLastExtendInto(kvs[lane], 0, seed, make([]float32, m.cfg.hidden), scalarWS); err != nil {
			b.Fatalf("seed lane %d: %v", lane, err)
		}
		tokens[lane] = 500 + lane
		hidden[lane] = make([]float32, m.cfg.hidden)
		logits[lane] = make([]float32, m.cfg.vocab)
		gotHidden[lane] = make([]float32, m.cfg.hidden)
	}

	wantHidden := make([][]float32, lanes)
	wantTokens := make([]int, lanes)
	if err := ws.gpu.decodeModeTile(m, kvs, tokens, hidden, logits, true, 4); err != nil {
		b.Fatalf("full-logit reference: %v", err)
	}
	for lane := range lanes {
		wantHidden[lane] = append([]float32(nil), hidden[lane]...)
		wantTokens[lane] = greedyArgmax(logits[lane])
	}
	variants := []struct {
		name string
		run  func() error
	}{
		{name: "full-copy-cpu-scan", run: func() error {
			if err := ws.gpu.decodeModeTile(m, kvs, tokens, hidden, logits, true, 4); err != nil {
				return err
			}
			for lane := range lanes {
				next[lane] = greedyArgmax(logits[lane])
			}
			return nil
		}},
		{name: "mapped-cpu-scan", run: func() error {
			return ws.gpu.decodeGreedyMode(m, kvs, tokens, gotHidden, next, true, 4, false)
		}},
		{name: "gpu-argmax", run: func() error {
			return ws.gpu.decodeGreedyMode(m, kvs, tokens, gotHidden, next, true, 4, true)
		}},
	}
	for _, variant := range variants {
		if err := variant.run(); err != nil {
			b.Fatalf("%s parity warmup: %v", variant.name, err)
		}
		for lane := range lanes {
			if next[lane] != wantTokens[lane] {
				b.Fatalf("%s lane %d token %d, want %d", variant.name, lane, next[lane], wantTokens[lane])
			}
			h := hidden[lane]
			if variant.name != "full-copy-cpu-scan" {
				h = gotHidden[lane]
			}
			for i := range h {
				if math.Float32bits(h[i]) != math.Float32bits(wantHidden[lane][i]) {
					b.Fatalf("%s lane %d hidden[%d] changed bits", variant.name, lane, i)
				}
			}
		}
	}

	elapsed := make([]time.Duration, len(variants))
	cpuElapsed := make([]time.Duration, len(variants))
	b.ReportAllocs()
	b.ResetTimer()
	iteration := 0
	for b.Loop() {
		for step := range variants {
			index := step
			if iteration%2 != 0 {
				index = len(variants) - 1 - step
			}
			cpuBefore := benchmarkProcessCPU(b)
			start := time.Now()
			err := variants[index].run()
			elapsed[index] += time.Since(start)
			cpuElapsed[index] += benchmarkProcessCPU(b) - cpuBefore
			if err != nil {
				b.Fatalf("%s decode: %v", variants[index].name, err)
			}
		}
		iteration++
	}
	b.StopTimer()
	for i, variant := range variants {
		b.ReportMetric(float64(elapsed[i].Nanoseconds())/float64(b.N)/1e6, variant.name+"-ms/step")
		b.ReportMetric(float64(cpuElapsed[i].Nanoseconds())/float64(b.N)/1e6, variant.name+"-cpu-ms/step")
	}
}

func benchmarkProcessCPU(b *testing.B) time.Duration {
	b.Helper()
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		b.Fatal(err)
	}
	return time.Duration(usage.Utime.Sec)*time.Second + time.Duration(usage.Utime.Usec)*time.Microsecond +
		time.Duration(usage.Stime.Sec)*time.Second + time.Duration(usage.Stime.Usec)*time.Microsecond
}
