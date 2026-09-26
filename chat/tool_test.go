// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package chat

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestFunc(t *testing.T) {
	type place struct {
		City string `json:"city"`
	}
	tool := Func("book", "Book a table.", func(_ context.Context, args struct {
		Name   string   `json:"name" desc:"who the table is for"`
		People int      `json:"people"`
		Time   string   `json:"time,omitempty" enum:"lunch, dinner"`
		Where  *place   `json:"where,omitempty"`
		Notes  []string `json:"notes,omitempty"`
		hidden int
	}) (string, error) {
		return args.Name + " " + strings.Repeat("x", args.People) + " " + args.Time + " " + args.Where.City + " " + strings.Join(args.Notes, "+"), nil
	})
	want := `{"type": "object", "properties": {` +
		`"name": {"type": "string", "description": "who the table is for"}, ` +
		`"people": {"type": "integer"}, ` +
		`"time": {"type": "string", "enum": ["lunch", "dinner"]}, ` +
		`"where": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}, ` +
		`"notes": {"type": "array", "items": {"type": "string"}}}, ` +
		`"required": ["name", "people"]}`
	if spec := tool.Spec(); spec.Name != "book" || spec.Description != "Book a table." || spec.Parameters != want {
		t.Fatalf("spec %+v\nwant parameters\n%s", spec, want)
	}
	got, err := tool.Call(context.Background(), []byte(`{"name": "Ana", "people": 3, "time": "dinner", "where": {"city": "Porto"}, "notes": ["a", "b"]}`))
	if err != nil || got != "Ana xxx dinner Porto a+b" {
		t.Fatalf("%q %v", got, err)
	}
	if _, err := tool.Call(context.Background(), []byte(`{"people": "many"}`)); err == nil {
		t.Fatal("bad arguments ran")
	}
	if specs := Specs([]Tool{tool}); len(specs) != 1 || specs[0].Name != "book" {
		t.Fatalf("specs %v", specs)
	}
}

// scripted is a Session that replies from a script: each reply is either
// words or a call.
type scripted struct {
	replies  []string // a reply starting with "call:" is a call to that tool
	messages []string
	calls    []Call
}

func (s *scripted) Add(role Role, text string) error {
	s.messages = append(s.messages, text)
	return nil
}
func (s *scripted) Reply(_ context.Context, _ Options, w io.Writer) error {
	s.calls = s.calls[:0]
	r := s.replies[0]
	s.replies = s.replies[1:]
	if name, ok := strings.CutPrefix(r, "call:"); ok {
		s.calls = append(s.calls, Call{Name: name, Arguments: []byte(`{}`)})
		return nil
	}
	_, err := w.Write([]byte(r))
	s.messages = append(s.messages, r)
	return err
}
func (s *scripted) Calls() []Call                                           { return s.calls }
func (s *scripted) Finished(context.Context, Role, string) (float32, error) { return 1, nil }
func (s *scripted) Prefill(context.Context) error                           { return nil }
func (s *scripted) Truncate(int) error                                      { return nil }
func (s *scripted) Checkpoint() int                                         { return len(s.messages) }
func (s *scripted) Restore(mark int) error                                  { s.messages = s.messages[:mark]; return nil }
func (s *scripted) Close() error                                            { return nil }

// Answer runs the calls a reply makes and replies again with their
// results; an unknown tool and a failing one answer with errors.
func TestAnswer(t *testing.T) {
	now := Func("now", "The time.", func(context.Context, struct{}) (string, error) { return "noon", nil })
	broken := Func("broken", "Fails.", func(context.Context, struct{}) (string, error) { return "", errors.New("down") })
	s := &scripted{replies: []string{"call:now", "call:broken", "call:missing", "It is noon."}}
	var out strings.Builder
	if err := Answer(context.Background(), s, []Tool{now, broken}, Options{}, &out, 4); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(s.messages, "|"); got != "noon|error: down|error: there is no tool missing|It is noon." || out.String() != "It is noon." {
		t.Fatalf("conversation %q, said %q", got, out.String())
	}
	// Rounds bound the loop: the last reply's calls are left unrun.
	s = &scripted{replies: []string{"call:now", "call:now"}}
	if err := Answer(context.Background(), s, []Tool{now}, Options{}, io.Discard, 2); err != nil || len(s.Calls()) != 1 || len(s.messages) != 1 {
		t.Fatalf("%v; %d calls left, %q", err, len(s.Calls()), s.messages)
	}
}
