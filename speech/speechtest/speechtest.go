// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package speechtest tests implementations of package speech's interfaces
// and the code that drives them.
package speechtest

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/gophonic/speech"
)

// Tone is a speech.Synthesizer for tests of what drives one: it speaks
// each byte of text as a fixed span of a constant level, as soon as the
// byte is written, so its audio and its Voiced are exact.
type Tone struct {
	perByte int
	level   float32

	mu      sync.Mutex
	id      uint64
	ctx     context.Context
	written int  // bytes of text
	ended   bool // End was called
	read    int  // samples read
	closed  bool
	wrote   chan struct{}
}

var _ speech.Synthesizer = (*Tone)(nil)

// ToneRate is a Tone's sample rate.
const ToneRate = 24000

// NewTone returns a Tone that speaks each byte of text for perByte, at
// level 0.5.
func NewTone(perByte time.Duration) *Tone {
	return &Tone{perByte: max(1, int(perByte*ToneRate/time.Second)), level: 0.5, wrote: make(chan struct{}, 1)}
}

// SampleRate is ToneRate.
func (t *Tone) SampleRate() int { return ToneRate }

// Voices lists none: any voice will do.
func (t *Tone) Voices() []string { return nil }

// Begin starts an utterance, dropping the one in progress.
func (t *Tone) Begin(ctx context.Context, _ speech.SpeakOptions) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return speech.ErrClosed
	}
	t.id++
	t.ctx, t.written, t.ended, t.read = ctx, 0, false, 0
	return nil
}

// Write adds text to the utterance.
func (t *Tone) Write(text []byte) (int, error) {
	t.mu.Lock()
	err := t.writable()
	if err == nil {
		t.written += len(text)
	}
	t.mu.Unlock()
	t.wake()
	if err != nil {
		return 0, err
	}
	return len(text), nil
}

// End marks the utterance's text complete.
func (t *Tone) End() error {
	t.mu.Lock()
	err := t.writable()
	t.ended = true
	t.mu.Unlock()
	t.wake()
	return err
}

func (t *Tone) writable() error {
	switch {
	case t.closed:
		return speech.ErrClosed
	case t.id == 0 || t.ended:
		return errors.New("speechtest: no utterance to write to")
	}
	return t.ctx.Err()
}

func (t *Tone) wake() {
	select {
	case t.wrote <- struct{}{}:
	default:
	}
}

// Read pulls the tone of the text written so far.
func (t *Tone) Read(pcm []float32) (int, error) {
	for {
		t.mu.Lock()
		switch {
		case t.closed:
			t.mu.Unlock()
			return 0, speech.ErrClosed
		case t.id == 0:
			t.mu.Unlock()
			return 0, errors.New("speechtest: no utterance to read")
		}
		ctx := t.ctx
		if err := ctx.Err(); err != nil {
			t.mu.Unlock()
			return 0, err
		}
		if len(pcm) == 0 {
			t.mu.Unlock()
			return 0, nil
		}
		if n := min(len(pcm), t.written*t.perByte-t.read); n > 0 {
			for i := range pcm[:n] {
				pcm[i] = t.level
			}
			t.read += n
			t.mu.Unlock()
			return n, nil
		}
		if t.ended {
			t.mu.Unlock()
			return 0, io.EOF
		}
		t.mu.Unlock()
		select {
		case <-t.wrote:
		case <-ctx.Done():
		}
	}
}

// Voiced reports the bytes whose tone has begun by the samples-th sample.
func (t *Tone) Voiced(samples int) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return min(t.written, (max(samples, 0)+t.perByte-1)/t.perByte)
}

// Close ends the lane.
func (t *Tone) Close() error {
	t.mu.Lock()
	t.closed = true
	t.mu.Unlock()
	t.wake()
	return nil
}

// TestSynthesizer checks that s keeps the speech.Synthesizer contract in
// opts's voice, speaking text (at least a few words), and closes it: an
// utterance whose text is written piece by piece while its audio is read,
// Voiced growing to all of the text, a Begin that drops the utterance in
// progress, a cancelled utterance, an empty one, and Synthesize.
func TestSynthesizer(t *testing.T, s speech.Synthesizer, opts speech.SpeakOptions, text string) {
	t.Helper()
	pcm := make([]float32, s.SampleRate()/20)
	ctx := context.Background()
	if _, err := s.Read(pcm); err == nil {
		t.Fatal("Read before Begin succeeded")
	}
	if _, err := s.Write([]byte(text)); err == nil {
		t.Fatal("Write before Begin succeeded")
	}

	// Text written as it is read, a word at a time.
	speak := func(text string) (samples int) {
		t.Helper()
		if err := s.Begin(ctx, opts); err != nil {
			t.Fatal(err)
		}
		done := make(chan error, 1)
		go func() {
			for _, w := range strings.SplitAfter(text, " ") {
				if _, err := s.Write([]byte(w)); err != nil {
					done <- err
					return
				}
			}
			done <- s.End()
		}()
		last := 0
		for {
			n, err := s.Read(pcm)
			samples += n
			v := s.Voiced(samples)
			if v < last || v > len(text) {
				t.Fatalf("Voiced %d after %d, of %d bytes", v, last, len(text))
			}
			last = v
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		if last = s.Voiced(samples); last != len(text) || samples == 0 {
			t.Fatalf("%d samples, Voiced %d at the end of %d bytes", samples, last, len(text))
		}
		if v0, vn := s.Voiced(0), s.Voiced(samples+1<<20); v0 != 0 || vn != len(text) {
			t.Fatalf("Voiced(0) = %d, Voiced past the end = %d, of %d bytes", v0, vn, len(text))
		}
		if _, err := s.Write([]byte("more")); err == nil {
			t.Fatal("Write after End succeeded")
		}
		return samples
	}
	whole := speak(text)

	// A Begin drops the utterance in progress: the next is spoken whole,
	// and only it.
	if err := s.Begin(ctx, opts); err != nil {
		t.Fatal(err)
	}
	s.Write([]byte(text))
	s.End()
	if _, err := s.Read(pcm); err != nil {
		t.Fatal(err)
	}
	short := text[:strings.IndexByte(text+" ", ' ')]
	if n := speak(short); n >= whole {
		t.Fatalf("%d samples for %q after a dropped utterance, %d for %q", n, short, whole, text)
	}

	// A cancelled utterance ends in its context's error.
	cut, cancel := context.WithCancel(ctx)
	if err := s.Begin(cut, opts); err != nil {
		t.Fatal(err)
	}
	s.Write([]byte(text))
	if _, err := s.Read(pcm); err != nil {
		t.Fatal(err)
	}
	cancel()
	if _, err := s.Read(pcm); !errors.Is(err, context.Canceled) {
		t.Fatalf("Read after cancel: %v", err)
	}
	if _, err := s.Write([]byte(text)); !errors.Is(err, context.Canceled) {
		t.Fatalf("Write after cancel: %v", err)
	}

	// An utterance without text says nothing.
	if err := s.Begin(ctx, opts); err != nil {
		t.Fatal(err)
	}
	s.End()
	if n, err := s.Read(pcm); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("an utterance without text: %d samples, %v", n, err)
	}

	out, err := speech.Synthesize(ctx, s, opts, text, nil)
	if err != nil || len(out) == 0 {
		t.Fatalf("Synthesize: %d samples, %v", len(out), err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("a second Close: %v", err)
	}
	if err := s.Begin(ctx, opts); err == nil {
		t.Fatal("Begin after Close succeeded")
	}
}
