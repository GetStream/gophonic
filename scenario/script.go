// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package scenario

import (
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/gophonic/speech"
)

// line is one step of a script.
type line struct {
	text string          // the line, for reports
	kind string          // user, wait, note, chat, join, leave, say, silent, speaks, says, repeats, captions
	arg  string          // the words, the claim, the name
	lang speech.Language // user(pt), gopher(pt)
	dur  time.Duration
}

// parse reads a script into its lines and its expectation.
func parse(script string) (lines []line, expected string, err error) {
	for n, raw := range strings.Split(script, "\n") {
		text := strings.TrimSpace(raw)
		if text == "" {
			continue
		}
		if strings.HasPrefix(text, "#") {
			if v, ok := strings.CutPrefix(strings.TrimSpace(text[1:]), "expect:"); ok {
				expected = strings.TrimSpace(v)
			}
			continue
		}
		who, rest, ok := strings.Cut(text, ":")
		if !ok {
			return nil, "", fmt.Errorf("line %d: %w: %q", n+1, errSyntax, text)
		}
		who, rest = strings.TrimSpace(who), strings.TrimSpace(rest)
		l := line{text: text}
		if i := strings.IndexByte(who, '('); i >= 0 && strings.HasSuffix(who, ")") {
			if l.lang, ok = speech.ParseLanguage(who[i+1 : len(who)-1]); !ok {
				return nil, "", fmt.Errorf("line %d: %w: unknown language in %q", n+1, errSyntax, who)
			}
			who = who[:i]
		}
		switch who {
		case "user":
			l.kind, l.arg = "user", rest
		case "wait":
			l.kind = "wait"
			if l.dur, err = time.ParseDuration(rest); err != nil {
				return nil, "", fmt.Errorf("line %d: %w", n+1, err)
			}
		case "note", "chat", "join", "leave", "say":
			l.kind, l.arg = who, rest
		default:
			// The agent's line: an assertion.
			if err := l.assertion(rest); err != nil {
				return nil, "", fmt.Errorf("line %d: %w", n+1, err)
			}
		}
		lines = append(lines, l)
	}
	return lines, expected, nil
}

// assertion reads the agent's line.
func (l *line) assertion(rest string) (err error) {
	word, after, _ := strings.Cut(rest, " ")
	switch strings.TrimRight(word, ",:") {
	case "silent":
		l.kind = "silent"
		if d, ok := strings.CutPrefix(after, "for "); ok {
			l.dur, err = time.ParseDuration(strings.TrimSpace(d))
		}
	case "speaks":
		l.kind = "speaks"
		if d, ok := strings.CutPrefix(after, "within "); ok {
			l.dur, err = time.ParseDuration(strings.TrimSpace(d))
		}
	case "says":
		l.kind, l.arg = "says", strings.TrimSpace(after)
	case "does":
		if strings.TrimSpace(after) != "not repeat the user" {
			return fmt.Errorf("%w: %q", errSyntax, rest)
		}
		l.kind = "repeats"
	case "captions":
		if strings.TrimSpace(after) != "follow the voice" {
			return fmt.Errorf("%w: %q", errSyntax, rest)
		}
		l.kind = "captions"
	default:
		return fmt.Errorf("%w: %q", errSyntax, rest)
	}
	return err
}
