// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"io"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/internal/testmodels"
)

func TestCompleteUTF8(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want int
	}{
		{"", 0}, {"abc", 3}, {"é", 2}, {"\xc3", 0}, {"a\xe2\x82", 1}, {"a\xe2\x82\xac", 4}, {"\xf0\x9f\x98", 0}, {"\xf0\x9f\x98\x80", 4},
	} {
		if got := completeUTF8([]byte(tc.in)); got != tc.want {
			t.Errorf("completeUTF8(%q) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

func TestSamplerTopK(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	logits := make([]float32, 5000)
	for i := range logits {
		logits[i] = r.Float32()
	}
	var s sampler
	s.topK(logits, 50)
	sorted := slices.Clone(logits)
	slices.SortFunc(sorted, func(a, b float32) int { return -cmpFloat(a, b) })
	for i := range 50 {
		if s.prob[i] != sorted[i] || logits[s.idx[i]] != sorted[i] {
			t.Fatalf("rank %d: got %v at %d, want %v", i, s.prob[i], s.idx[i], sorted[i])
		}
	}
	// With a temperature, draws come only from the top k and follow the
	// softmax: the best of two logits apart by ln 3 is drawn about 3:1.
	two := []float32{0, 1.0986123, -50}
	counts := [3]int{}
	s.reset(chat.Options{Seed: 7})
	for range 40000 {
		counts[s.next(two, chat.Options{Temperature: 1, TopK: 2})]++
	}
	if counts[2] != 0 || counts[1] < 29000 || counts[1] > 31000 {
		t.Fatalf("draws %v, want about 10000:30000:0", counts)
	}
}

func cmpFloat(a, b float32) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

func TestChatOfficial(t *testing.T) {
	g, err := Open(testmodels.Path(t, testmodels.Qwen3), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	s, err := g.NewSession("Answer with one word.")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	reply := func(opts chat.Options) string {
		var b strings.Builder
		if err := s.Reply(context.Background(), opts, &b); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	s.Add(chat.User, "What is the capital of France?")
	if got := reply(chat.Options{}); !strings.Contains(got, "Paris") {
		t.Fatalf("reply %q", got)
	}
	// Truncating the reply rewrites the conversation: the next reply sees
	// only what was kept.
	if err := s.Truncate(0); err != nil {
		t.Fatal(err)
	}
	s.Add(chat.User, "What is the capital of Italy?")
	if got := reply(chat.Options{}); !strings.Contains(got, "Rome") {
		t.Fatalf("reply %q", got)
	}
	// Sampling with a seed is repeatable.
	var seeded [2]string
	for i := range seeded {
		other, err := g.NewSession("Answer with one word.")
		if err != nil {
			t.Fatal(err)
		}
		other.Add(chat.User, "Name a color.")
		var b strings.Builder
		if err := other.Reply(context.Background(), chat.Options{Temperature: 0.8, TopP: 0.95, Seed: 42, MaxTokens: 8}, &b); err != nil {
			t.Fatal(err)
		}
		other.Close()
		seeded[i] = b.String()
	}
	if seeded[0] != seeded[1] || seeded[0] == "" {
		t.Fatalf("seeded replies %q and %q", seeded[0], seeded[1])
	}
	// A transcript prefilled while it grows, then replaced by the whole
	// one, changes nothing but how soon the reply starts.
	var replies [4]string
	var first [4]time.Duration
	var finished, again float32
	for i := range replies {
		other, err := g.NewSession("Answer in one short sentence.")
		if err != nil {
			t.Fatal(err)
		}
		if err := other.Prefill(context.Background()); err != nil {
			t.Fatal(err)
		}
		if i == 1 {
			for _, partial := range []string{"What is", "What is the tallest", "What is the tallest mountain in"} {
				mark := other.Checkpoint()
				other.Add(chat.User, partial)
				if err := other.Prefill(context.Background()); err != nil {
					t.Fatal(err)
				}
				if err := other.Restore(mark); err != nil {
					t.Fatal(err)
				}
			}
		}
		question := "What is the tallest mountain in Europe?"
		if i == 3 {
			// Judging the words as they are heard, the last time as they
			// end, leaves nothing for the answer's judgment to evaluate.
			for _, partial := range []string{"What is", "What is the tallest mountain", question} {
				if finished, err = other.Finished(context.Background(), chat.User, partial); err != nil {
					t.Fatal(err)
				}
			}
		}
		began := time.Now()
		if i == 2 {
			// Judging whether the words are finished evaluates the start of
			// the reply as well.
			if finished, err = other.Finished(context.Background(), chat.User, question); err != nil {
				t.Fatal(err)
			}
		}
		if i == 3 {
			if again, err = other.Finished(context.Background(), chat.User, question); err != nil {
				t.Fatal(err)
			}
			if again != finished {
				t.Fatalf("the same words judged again: %v, then %v", finished, again)
			}
		}
		other.Add(chat.User, question)
		b := timedWriter{began: began, first: &first[i]}
		if err := other.Reply(context.Background(), chat.Options{Temperature: 0.7, Seed: 3, MaxTokens: 24}, &b); err != nil {
			t.Fatal(err)
		}
		other.Close()
		replies[i] = b.String()
	}
	if replies[0] != replies[1] || replies[0] != replies[2] || replies[0] != replies[3] {
		t.Fatalf("plain reply %q, prefilled %q, after Finished %q, after judging it as heard %q", replies[0], replies[1], replies[2], replies[3])
	}
	if finished < 0.01 {
		t.Fatalf("a whole question ends with probability %g", finished)
	}
	t.Logf("first text after %v, %v when prefilled, %v after Finished (%.3f), %v when judged as heard: %q", first[0], first[1], first[2], finished, first[3], replies[1])
	s.Add(chat.User, "Say hi.")
	allocs := testing.AllocsPerRun(3, func() {
		if err := s.Reply(context.Background(), chat.Options{Temperature: 0.7, TopK: 40, Presence: 1.5, MaxTokens: 16}, io.Discard); err != nil {
			t.Fatal(err)
		}
		s.Add(chat.User, "Again.")
	})
	if allocs != 0 {
		t.Fatalf("warm replies allocate %v times", allocs)
	}
}

// timedWriter collects a reply and notes when its first piece arrived.
type timedWriter struct {
	strings.Builder
	began time.Time
	first *time.Duration
}

func (w *timedWriter) Write(p []byte) (int, error) {
	if w.Len() == 0 {
		*w.first = time.Since(w.began)
	}
	return w.Builder.Write(p)
}

// A model prepares each capability at its first use: nothing but the
// weights at Open, the question workspace for a question, and the
// language-model head for a conversation.
func TestModelPreparesOnUse(t *testing.T) {
	m, err := Open(testmodels.Path(t, testmodels.Qwen3), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.weights.HasHead() || m.enc != nil || m.gen != nil {
		t.Fatal("Open prepared more than the weights")
	}
	q, err := m.Question("Is this a greeting?", []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	probs := make([]float32, 2)
	if err := q.Choose(context.Background(), "Hello there!", probs); err != nil || probs[0] < probs[1] {
		t.Fatalf("a greeting: %v, %v", probs, err)
	}
	if m.weights.HasHead() || m.gen != nil {
		t.Fatal("a question loaded the head")
	}
	s, err := m.NewSession("Answer with one word.")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if !m.weights.HasHead() {
		t.Fatal("a conversation without the head")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.NewSession(""); err == nil {
		t.Fatal("a session of a closed model")
	}
}
