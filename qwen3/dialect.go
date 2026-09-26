// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"

	"github.com/GetStream/gophonic/chat"
	"github.com/thesyncim/vibejson"
)

// A dialect is how a model family was trained to call tools. Both offer the
// tools' signatures as JSON in the system message, write each call between
// <tool_call> tokens, and take each result in a user message between
// <tool_response> tags; they differ in how a call is written.
type dialect uint8

const (
	// jsonCalls is Qwen3's: {"name": ..., "arguments": {...}}.
	jsonCalls dialect = iota
	// xmlCalls is the Qwen3.5 family's (Qwen3.6): <function=name>, then
	// <parameter=p> blocks holding strings as they are and other values as
	// JSON, with the system prompt after the tools.
	xmlCalls
)

// chatTemplate returns the chat template of the snapshot at path, or "".
func chatTemplate(path string) string {
	if raw, err := os.ReadFile(filepath.Join(path, "tokenizer_config.json")); err == nil {
		var cfg struct {
			Template string `json:"chat_template"`
		}
		if vibejson.Unmarshal(raw, &cfg) == nil && cfg.Template != "" {
			return cfg.Template
		}
	}
	raw, _ := os.ReadFile(filepath.Join(path, "chat_template.jinja"))
	return string(raw)
}

// dialectOf returns the dialect of the snapshot at path's chat template.
func dialectOf(path string) dialect {
	if strings.Contains(chatTemplate(path), "<function=") {
		return xmlCalls
	}
	return jsonCalls
}

// prompt returns the system message that offers tools, as the family's
// chat template renders it.
func (d dialect) prompt(system string, tools []chat.ToolSpec) (string, error) {
	var b strings.Builder
	if d == jsonCalls && system != "" {
		b.WriteString(system)
		b.WriteString("\n\n")
	}
	if d == jsonCalls {
		b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
			"You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
	} else {
		b.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
	}
	for _, t := range tools {
		if err := writeTool(&b, t); err != nil {
			return "", err
		}
	}
	if d == jsonCalls {
		b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments " +
			"within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
		return b.String(), nil
	}
	b.WriteString("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
		"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
		"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n" +
		"</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
		"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
		"- Required parameters MUST be specified\n" +
		"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
		"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
		"</IMPORTANT>")
	if system = strings.TrimSpace(system); system != "" {
		b.WriteString("\n\n")
		b.WriteString(system)
	}
	return b.String(), nil
}

// scaffold is what follows <tool_call> in a call to name, up to its
// arguments, which a session drafts whole.
func (d dialect) scaffold(name string) string {
	if d == xmlCalls {
		return "\n<function=" + name + ">\n"
	}
	return "\n{\"name\": \"" + name + "\", \"arguments\":"
}

// parse reads a call written between <tool_call> tokens, returning its name
// and its arguments as a JSON object appended to dst. strings reports the
// parameters a tool takes as strings, whose XML values are raw text.
func (d dialect) parse(call []byte, strings func(name, param []byte) bool, dst []byte) (name, args []byte, err error) {
	if d == jsonCalls {
		name, args, err = parseCall(call)
		return name, append(dst, args...), err
	}
	return parseXMLCall(call, strings, dst)
}

// parseXMLCall reads <function=name> and its <parameter=p> blocks.
func parseXMLCall(b []byte, isString func(name, param []byte) bool, dst []byte) (name, args []byte, err error) {
	_, rest, ok := bytes.Cut(b, []byte("<function="))
	if !ok {
		return nil, nil, errCall
	}
	name, rest, ok = bytes.Cut(rest, []byte(">"))
	if !ok || len(name) == 0 {
		return nil, nil, errCall
	}
	dst = append(dst, '{')
	first := true
	for {
		var param, value []byte
		if _, rest, ok = bytes.Cut(rest, []byte("<parameter=")); !ok {
			break
		}
		if param, rest, ok = bytes.Cut(rest, []byte(">")); !ok {
			return nil, nil, errCall
		}
		if value, rest, ok = bytes.Cut(rest, []byte("</parameter>")); !ok {
			return nil, nil, errCall
		}
		// The template puts a newline after the tag and before the end.
		value = bytes.TrimPrefix(value, []byte("\n"))
		value = bytes.TrimSuffix(value, []byte("\n"))
		if !first {
			dst = append(dst, ',')
		}
		first = false
		dst = appendJSONString(dst, param)
		dst = append(dst, ':')
		if isString(name, param) || !vibejson.Valid(value) {
			dst = appendJSONString(dst, value)
		} else {
			dst = append(dst, value...)
		}
	}
	return name, append(dst, '}'), nil
}

// appendJSONString appends s as a JSON string.
func appendJSONString(dst, s []byte) []byte {
	const hex = "0123456789abcdef"
	dst = append(dst, '"')
	for _, c := range s {
		switch {
		case c == '"' || c == '\\':
			dst = append(dst, '\\', c)
		case c == '\n':
			dst = append(dst, '\\', 'n')
		case c == '\t':
			dst = append(dst, '\\', 't')
		case c == '\r':
			dst = append(dst, '\\', 'r')
		case c < 0x20:
			dst = append(dst, '\\', 'u', '0', '0', hex[c>>4], hex[c&15])
		default:
			dst = append(dst, c)
		}
	}
	return append(dst, '"')
}

// stringParams returns the parameters of a tool's JSON schema whose type is
// string.
func stringParams(schema string) []string {
	var s struct {
		Properties map[string]struct {
			Type string `json:"type"`
		} `json:"properties"`
	}
	if schema == "" || vibejson.Unmarshal([]byte(schema), &s) != nil {
		return nil
	}
	var names []string
	for name, p := range s.Properties {
		if p.Type == "string" {
			names = append(names, name)
		}
	}
	return names
}
