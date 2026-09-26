// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/internal/testmodels"
)

var clock = chat.ToolSpec{
	Name:        "now",
	Description: "The current date and time.",
	Parameters:  `{"type":"object","properties":{"zone":{"type":"string","description":"an IANA time zone, such as Europe/Lisbon"},"seconds":{"type":"boolean"}}}`,
}

// Qwen3.5-family calls read as JSON arguments: strings as written, other
// values as the JSON they are.
func TestParseXMLCall(t *testing.T) {
	call := []byte("\n<function=now>\n<parameter=zone>\nAsia/Tokyo \"JST\"\n</parameter>\n<parameter=seconds>\ntrue\n</parameter>\n" +
		"<parameter=note>\ntwo\nlines\n</parameter>\n</function>\n")
	strs := func(name, param []byte) bool { return string(param) == "zone" }
	name, args, err := xmlCalls.parse(call, strs, nil)
	if err != nil || string(name) != "now" || string(args) != `{"zone":"Asia/Tokyo \"JST\"","seconds":true,"note":"two\nlines"}` {
		t.Fatalf("%q %s %v", name, args, err)
	}
	if _, _, err := xmlCalls.parse([]byte("<parameter=x>\n1\n</parameter>"), strs, nil); err == nil {
		t.Fatal("a call without a function parsed")
	}
	if got := stringParams(clock.Parameters); len(got) != 1 || got[0] != "zone" {
		t.Fatalf("string parameters %v", got)
	}
}

// The Qwen3.5 family's template lists the tools first, then the system
// prompt.
func TestXMLToolsPrompt(t *testing.T) {
	p, err := xmlCalls.prompt("  Be brief.\n", []chat.ToolSpec{clock})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(p, "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"now\"") ||
		!strings.HasSuffix(p, "</IMPORTANT>\n\nBe brief.") {
		t.Fatalf("prompt %q", p)
	}
}

// Qwen3.6-35B-A3B chats: it answers, calls tools in its own format (drafted
// calls equal to decoding token by token), and answers zero-shot questions.
func TestChatQwen36(t *testing.T) {
	path := testmodels.Path(t, testmodels.Qwen36)
	if !IsModelDir(path) {
		t.Fatal("not recognized as a Qwen3 checkpoint")
	}
	g, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if gen, err := g.generator(); err != nil || gen.dialect != xmlCalls {
		t.Fatalf("the Qwen3.5 tool dialect was not chosen: %v", err)
	}
	s, err := g.NewSession("You are a voice assistant. Answer in one short sentence.")
	if err != nil {
		t.Fatal(err)
	}
	s.Add(chat.User, "What is the capital of France?")
	var b strings.Builder
	if err := s.Reply(context.Background(), chat.Options{MaxTokens: 40}, &b); err != nil {
		t.Fatal(err)
	}
	t.Logf("reply: %q", b.String())
	if !strings.Contains(b.String(), "Paris") {
		t.Errorf("reply %q", b.String())
	}
	s.Close()

	var args [2]string
	var took [2]time.Duration
	for i := range args {
		sess, err := g.NewSession("You are a voice assistant. Use your tools rather than guessing.", clock)
		if err != nil {
			t.Fatal(err)
		}
		s := sess.(*session)
		if i == 1 {
			s.tools[0].scaffold = nil // no drafts
		}
		s.Add(chat.User, "What time is it in Tokyo?")
		b.Reset()
		began := time.Now()
		if err := s.Reply(context.Background(), chat.Options{MaxTokens: 80}, &b); err != nil {
			t.Fatal(err)
		}
		took[i] = time.Since(began)
		calls := s.Calls()
		if len(calls) != 1 || calls[0].Name != "now" || !strings.Contains(string(calls[0].Arguments), "Asia/Tokyo") {
			t.Fatalf("text %q, calls %+v", b.String(), calls)
		}
		if strings.Contains(b.String(), "<function") {
			t.Errorf("the call reached the text: %q", b.String())
		}
		args[i] = string(calls[0].Arguments)
		s.Add(chat.ToolResult, `{"time": "21:04", "zone": "Asia/Tokyo"}`)
		b.Reset()
		if err := s.Reply(context.Background(), chat.Options{MaxTokens: 40}, &b); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(b.String(), "9:04") && !strings.Contains(b.String(), "21:04") {
			t.Errorf("answer after the result: %q", b.String())
		}
		s.Close()
	}
	if args[0] != args[1] {
		t.Errorf("drafted %s, decoded %s", args[0], args[1])
	}
	t.Logf("call %s: %v drafted, %v token by token", args[0], took[0].Round(time.Millisecond), took[1].Round(time.Millisecond))

	// The same model answers questions too.
	q, err := g.Question("Is the user greeting the assistant?", []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	probs := make([]float32, 2)
	for _, c := range []struct {
		text string
		yes  bool
	}{{"Hello there, how are you?", true}, {"What is the capital of France?", false}} {
		if err := q.Choose(context.Background(), c.text, probs); err != nil {
			t.Fatal(err)
		}
		t.Logf("%q: yes %.2f", c.text, probs[0])
		if (probs[0] > 0.5) != c.yes {
			t.Errorf("%q: yes %.2f", c.text, probs[0])
		}
	}
}
