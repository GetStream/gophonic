// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3asr

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

// BatchTranscriber owns persistent private transcription lanes and a decoder
// coordinator. Compatible next-token projections reuse immutable GPU weights
// across lanes, with FP32 activations and each lane's own K/V cache. Prefills
// and audio encoding still use each lane's ordinary transcriber.
//
// Transcribe calls on one BatchTranscriber must not overlap. The model must
// outlive it. A batch advances at decode boundaries: a slower lane can delay
// other lanes, so use Transcriber when independent call latency matters most.
type BatchTranscriber struct {
	m        *Model
	decode   *qwen3lm.DecodeBatchWorkspace
	lanes    []batchWorker
	messages chan batchMessage
	stopped  sync.WaitGroup
	closed   bool

	pending [8]int
	prefix  [8]*qwen3lm.PrefixKV
	tokens  [8]int
	hidden  [8][]float32
	next    [8]int
	logits  [8][]float32
}

type batchWorker struct {
	tr       *Transcriber
	jobs     chan batchInput
	step     batchDecodeLane
	closeErr error
}

type batchInput struct {
	ctx   context.Context
	pcm   []float32
	opts  speech.Options
	dst   *speech.Transcript
	batch bool
}

type batchMessage struct {
	lane int
	done bool
	err  error
}

// Only messages cross lane boundaries. Sending a token request transfers
// exclusive use of that lane's decoder buffers until its reply arrives.
type batchDecodeLane struct {
	index    int
	messages chan<- batchMessage
	reply    chan error
}

func (lane *batchDecodeLane) advance(ctx context.Context) error {
	select {
	case lane.messages <- batchMessage{lane: lane.index}:
		// Even after cancellation, wait for ownership to return before the
		// lane can reset or release its buffers.
		return <-lane.reply
	case <-ctx.Done():
		return ctx.Err()
	}
}

// NewBatchTranscriber opens capacity private lanes (1–8), configured by opts.
// The initial batched decoder supports "gpu-q8" (Q8B Metal); unsupported
// formats return an error without changing precision.
// A batch containing only one call uses the ordinary scalar decode path.
func NewBatchTranscriber(m *Model, capacity int, opts LaneOptions) (*BatchTranscriber, error) {
	if m == nil || m.enc == nil || m.eval == nil || m.lm == nil {
		return nil, errors.New("qwen3asr: nil or closed model")
	}
	if capacity < 1 || capacity > 8 {
		return nil, fmt.Errorf("qwen3asr: batch capacity %d outside [1,8]", capacity)
	}
	if opts.Threads < 0 || opts.Threads > 64 {
		return nil, fmt.Errorf("qwen3asr: invalid thread count %d", opts.Threads)
	}
	decode, err := m.eval.NewDecodeBatchWorkspace(capacity)
	if err != nil {
		return nil, fmt.Errorf("qwen3asr: batch decoder: %w", err)
	}
	b := &BatchTranscriber{
		m: m, decode: decode, lanes: make([]batchWorker, capacity),
		messages: make(chan batchMessage, capacity),
	}
	for i := range b.lanes {
		lane := &b.lanes[i]
		lane.tr, err = NewTranscriber(m, opts)
		if err != nil {
			for j := range i {
				_ = b.lanes[j].tr.Close()
			}
			_ = decode.Close()
			return nil, err
		}
		lane.jobs = make(chan batchInput, 1)
		lane.step = batchDecodeLane{index: i, messages: b.messages, reply: make(chan error, 1)}
	}
	for i := range b.lanes {
		lane := &b.lanes[i]
		b.stopped.Add(1)
		go func() {
			defer b.stopped.Done()
			defer func() { lane.closeErr = lane.tr.Close() }()
			for job := range lane.jobs {
				if job.batch {
					lane.tr.decodeBatch = &lane.step
				}
				err := lane.tr.Transcribe(job.ctx, job.pcm, job.opts, job.dst)
				lane.tr.decodeBatch = nil
				b.messages <- batchMessage{lane: lane.step.index, done: true, err: err}
			}
		}()
	}
	return b, nil
}

// Transcribe writes one transcript per PCM input. opts may be nil for default
// options, or contain one Options value per input. dst must have the same
// length as pcm. Inputs and partial transcripts must remain immutable until
// the call returns, including after cancellation. Each output must own its
// storage independently; outputs cannot alias inputs or partial transcripts.
// Empty inputs, chunking, languages, context, segments, and partial transcripts
// follow Transcriber.Transcribe's behavior. On error, some outputs may be
// partial; every worker has returned before this method returns.
func (b *BatchTranscriber) Transcribe(ctx context.Context, pcm [][]float32, opts []speech.Options, dst []speech.Transcript) error {
	if b == nil || b.closed {
		return speech.ErrClosed
	}
	count := len(pcm)
	if count > len(b.lanes) || len(dst) != count || len(opts) != 0 && len(opts) != count {
		return fmt.Errorf("qwen3asr: %d inputs, %d options, %d outputs for batch capacity %d", count, len(opts), len(dst), len(b.lanes))
	}
	if ctx == nil {
		return errors.New("qwen3asr: nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Reset preserves slice capacity, so check writable backing ranges even
	// when a previous output's current length is zero.
	for i := range dst {
		for j := 0; j < i; j++ {
			if batchTranscriptOverlap(&dst[i], &dst[j]) {
				return errors.New("qwen3asr: batch outputs have overlapping storage")
			}
		}
		for _, opt := range opts {
			if opt.Partial != nil && (opt.Partial == &dst[i] || batchTranscriptOverlap(opt.Partial, &dst[i])) {
				return errors.New("qwen3asr: partial transcript aliases a batch output")
			}
		}
	}
	for i := range count {
		var opt speech.Options
		if len(opts) != 0 {
			opt = opts[i]
		}
		b.lanes[i].jobs <- batchInput{ctx: ctx, pcm: pcm[i], opts: opt, dst: &dst[i], batch: count > 1}
	}
	var first error
	pending, alive := 0, count
	cancelled := ctx.Done()
	for alive > 0 {
		select {
		case message := <-b.messages:
			if message.done {
				alive--
				if first == nil {
					first = message.err
				}
			} else {
				b.pending[pending] = message.lane
				pending++
			}
		case <-cancelled:
			if first == nil {
				first = ctx.Err()
			}
			cancelled = nil
		}
		if pending == 0 || pending < alive && first == nil {
			continue
		}
		if first == nil && pending == 1 {
			// A lone remaining lane cannot reuse weights across calls. Its
			// private scalar workspace avoids batch barriers for the tail.
			tr := b.lanes[b.pending[0]].tr
			first = b.m.eval.HiddenLastExtendInto(tr.kv, len(tr.kv.Tokens()), tr.gen[len(tr.gen)-1:], tr.hidden, tr.lm)
			if first == nil {
				first = b.m.eval.LogitsInto(tr.hidden, tr.logits, tr.lm)
				tr.nextToken = tr.pick(tr.logits, tr.gen)
			}
		} else if first == nil {
			constrained := false
			for row, index := range b.pending[:pending] {
				tr := b.lanes[index].tr
				b.prefix[row], b.tokens[row] = tr.kv, tr.gen[len(tr.gen)-1]
				b.hidden[row], b.logits[row] = tr.hidden, tr.logits
				constrained = constrained || tr.limits != nil
			}
			// Language and script constraints need their allowed-score sets.
			// Preserve that selection exactly; unrestricted batches return only
			// token IDs and avoid vocabulary readback.
			if constrained {
				first = b.m.eval.DecodeBatchInto(b.prefix[:pending], b.tokens[:pending], b.hidden[:pending], b.logits[:pending], b.decode)
				if first == nil {
					for row, index := range b.pending[:pending] {
						tr := b.lanes[index].tr
						b.next[row] = tr.pick(tr.logits, tr.gen)
					}
				}
			} else {
				first = b.m.eval.DecodeBatchGreedyInto(b.prefix[:pending], b.tokens[:pending], b.hidden[:pending], b.next[:pending], b.decode)
			}
			if first == nil {
				for row, index := range b.pending[:pending] {
					b.lanes[index].tr.nextToken = b.next[row]
				}
			}
			clear(b.prefix[:pending])
			clear(b.hidden[:pending])
			clear(b.logits[:pending])
		}
		for _, index := range b.pending[:pending] {
			b.lanes[index].step.reply <- first
		}
		pending = 0
	}
	return first
}

func batchTranscriptOverlap(a, b *speech.Transcript) bool {
	return batchSliceOverlap(a.Text, b.Text) || batchSliceOverlap(a.Segments, b.Segments) || batchSliceOverlap(a.Words, b.Words)
}

func batchSliceOverlap[T any](a, b []T) bool {
	if cap(a) == 0 || cap(b) == 0 {
		return false
	}
	var element T
	width := unsafe.Sizeof(element)
	a0, b0 := uintptr(unsafe.Pointer(unsafe.SliceData(a))), uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return a0 < b0+uintptr(cap(b))*width && b0 < a0+uintptr(cap(a))*width
}

// Close releases all private lanes and decoder scratch. It must not overlap
// Transcribe. Repeated calls are harmless.
func (b *BatchTranscriber) Close() error {
	if b == nil || b.closed {
		return nil
	}
	b.closed = true
	for i := range b.lanes {
		close(b.lanes[i].jobs)
	}
	b.stopped.Wait()
	first := b.decode.Close()
	for i := range b.lanes {
		if first == nil {
			first = b.lanes[i].closeErr
		}
	}
	// Workers have exited, so retire all references without copying the
	// WaitGroup, which must not be copied after use.
	b.m, b.decode, b.lanes, b.messages = nil, nil, nil, nil
	clear(b.prefix[:])
	clear(b.hidden[:])
	clear(b.logits[:])
	return first
}
