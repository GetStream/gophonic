// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/gophonic/chat"
	"github.com/thesyncim/vibejson"
)

// Tool calls as Qwen3 was trained to make them: the tools' signatures as
// JSON in the system message, each call as {"name": ..., "arguments": ...}
// between <tool_call> tokens, and each result in a user message between
// <tool_response> tags.

// toolsPrompt returns the system message that offers tools, as Qwen3's chat
// template renders it.
func toolsPrompt(system string, tools []chat.ToolSpec) (string, error) {
	var b strings.Builder
	if system != "" {
		b.WriteString(system)
		b.WriteString("\n\n")
	}
	b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
		"You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
	for _, t := range tools {
		name, err := vibejson.Marshal(&t.Name)
		if err != nil {
			return "", err
		}
		description, err := vibejson.Marshal(&t.Description)
		if err != nil {
			return "", err
		}
		params := t.Parameters
		if params == "" {
			params = `{"type": "object", "properties": {}}`
		}
		spaced, err := pythonJSON([]byte(params))
		if err != nil {
			return "", fmt.Errorf("qwen3: tool %s: parameters: %w", t.Name, err)
		}
		b.WriteString("\n{\"type\": \"function\", \"function\": {\"name\": ")
		b.Write(name)
		b.WriteString(", \"description\": ")
		b.Write(description)
		b.WriteString(", \"parameters\": ")
		b.Write(spaced)
		b.WriteString("}}")
	}
	b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments " +
		"within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
	return b.String(), nil
}

// pythonJSON reformats JSON with the separators of Python's json.dumps,
// ", " and ": ", which the template's tojson writes.
func pythonJSON(src []byte) ([]byte, error) {
	compact, err := vibejson.Compact(src)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, len(compact)+len(compact)/4)
	quoted, escaped := false, false
	for _, c := range compact {
		out = append(out, c)
		switch {
		case escaped:
			escaped = false
		case quoted && c == '\\':
			escaped = true
		case c == '"':
			quoted = !quoted
		case !quoted && (c == ',' || c == ':'):
			out = append(out, ' ')
		}
	}
	return out, nil
}

var errCall = errors.New("qwen3: malformed tool call")

// parseCall reads a call's {"name": ..., "arguments": ...} object, returning
// the name's and the arguments' spans in b.
func parseCall(b []byte) (name, args []byte, err error) {
	i := skipSpace(b, 0)
	if i == len(b) || b[i] != '{' {
		return nil, nil, errCall
	}
	i++
	for {
		i = skipSpace(b, i)
		if i < len(b) && b[i] == '}' {
			break
		}
		kEnd, ok := stringEnd(b, i)
		if !ok {
			return nil, nil, errCall
		}
		key := b[i+1 : kEnd-1]
		i = skipSpace(b, kEnd)
		if i == len(b) || b[i] != ':' {
			return nil, nil, errCall
		}
		i = skipSpace(b, i+1)
		vEnd, ok := valueEnd(b, i)
		if !ok {
			return nil, nil, errCall
		}
		switch string(key) {
		case "name":
			if b[i] != '"' {
				return nil, nil, errCall
			}
			name = b[i+1 : vEnd-1]
		case "arguments":
			args = b[i:vEnd]
		}
		i = skipSpace(b, vEnd)
		if i < len(b) && b[i] == ',' {
			i++
		}
	}
	if name == nil {
		return nil, nil, errCall
	}
	if args == nil {
		args = []byte("{}")
	}
	return name, args, nil
}

func skipSpace(b []byte, i int) int {
	for i < len(b) && (b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r') {
		i++
	}
	return i
}

// stringEnd returns the index after the JSON string starting at b[i].
func stringEnd(b []byte, i int) (int, bool) {
	if i >= len(b) || b[i] != '"' {
		return 0, false
	}
	for j := i + 1; j < len(b); j++ {
		switch b[j] {
		case '\\':
			j++
		case '"':
			return j + 1, true
		}
	}
	return 0, false
}

// valueEnd returns the index after the JSON value starting at b[i].
func valueEnd(b []byte, i int) (int, bool) {
	if i >= len(b) {
		return 0, false
	}
	switch b[i] {
	case '"':
		return stringEnd(b, i)
	case '{', '[':
		depth := 0
		for j := i; j < len(b); j++ {
			switch b[j] {
			case '"':
				end, ok := stringEnd(b, j)
				if !ok {
					return 0, false
				}
				j = end - 1
			case '{', '[':
				depth++
			case '}', ']':
				if depth--; depth == 0 {
					return j + 1, true
				}
			}
		}
		return 0, false
	}
	j := i
	for j < len(b) && b[j] != ',' && b[j] != '}' && b[j] != ']' && b[j] != ' ' && b[j] != '\n' {
		j++
	}
	return j, j > i
}
