// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/internal/testmodels"
)

var quiet = chat.ToolSpec{
	Name:        "stay_quiet",
	Description: "Stop answering and only listen, until someone says a given word or phrase. Use it when asked to be quiet, to stop talking, or to wait.",
	Parameters:  `{"type":"object","properties":{"until":{"type":"string","description":"the word or phrase that ends the quiet, if one was named"}}}`,
}

func TestParseCall(t *testing.T) {
	name, args, err := parseCall([]byte("\n{\"name\": \"stay_quiet\", \"arguments\": {\"until\": \"hi \\\"there\\\"\", \"n\": [1, {\"a\": 2}]}}\n"))
	if err != nil || string(name) != "stay_quiet" || string(args) != `{"until": "hi \"there\"", "n": [1, {"a": 2}]}` {
		t.Fatalf("%q %q %v", name, args, err)
	}
	if _, _, err := parseCall([]byte(`{"arguments": {}}`)); err == nil {
		t.Fatal("a call without a name parsed")
	}
}

func TestToolsPrompt(t *testing.T) {
	p, err := toolsPrompt("Be brief.", []chat.ToolSpec{quiet})
	if err != nil {
		t.Fatal(err)
	}
	want := "Be brief.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
		"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
		`{"type": "function", "function": {"name": "stay_quiet", "description": "` + quiet.Description + `", ` +
		`"parameters": {"type": "object", "properties": {"until": {"type": "string", "description": "the word or phrase that ends the quiet, if one was named"}}}}}` +
		"\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n" +
		"<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
	if p != want {
		t.Fatalf("prompt\n%s\nwant\n%s", p, want)
	}
}

// Asked to be quiet, Qwen3-8B calls the tool rather than saying so; the
// call is not text, and drafting its scaffold changes nothing but speed.
func TestChatTools(t *testing.T) {
	g, err := OpenChat(testmodels.Path(t, testmodels.Qwen3), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	var text [2]string
	var args [2]string
	var took [2]time.Duration
	for i := range text {
		sess, err := g.NewSession("You are a voice assistant in a call. Answer in one short sentence.", quiet)
		if err != nil {
			t.Fatal(err)
		}
		s := sess.(*Session)
		if i == 1 {
			s.tools = nil // no drafts
		}
		s.Add(chat.User, "Hello!")
		s.Reply(context.Background(), chat.Options{Temperature: 0.7, Seed: 5, MaxTokens: 40}, io.Discard)
		s.Add(chat.User, "Please stop talking until I say the word hi.")
		var b strings.Builder
		began := time.Now()
		if err := s.Reply(context.Background(), chat.Options{Temperature: 0.7, Seed: 5, MaxTokens: 60}, &b); err != nil {
			t.Fatal(err)
		}
		took[i] = time.Since(began)
		text[i] = b.String()
		calls := s.Calls()
		if len(calls) != 1 || calls[0].Name != "stay_quiet" || !strings.Contains(strings.ToLower(string(calls[0].Arguments)), "hi") {
			t.Fatalf("calls %+v", calls)
		}
		args[i] = string(calls[0].Arguments)
		if strings.Contains(text[i], "tool_call") || strings.Contains(text[i], "{") {
			t.Errorf("the call reached the text: %q", text[i])
		}
		// The result joins the conversation, and the conversation goes on.
		if err := s.Add(chat.ToolResult, "Quiet until someone says \"hi\"."); err != nil {
			t.Fatal(err)
		}
		s.Close()
	}
	if text[0] != text[1] || args[0] != args[1] {
		t.Errorf("drafted %q %s, decoded %q %s", text[0], args[0], text[1], args[1])
	}
	t.Logf("call arguments %s: %v drafted, %v token by token", args[0], took[0].Round(time.Millisecond), took[1].Round(time.Millisecond))
}
