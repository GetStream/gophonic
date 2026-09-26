// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package duplex

import (
	"context"

	"github.com/GetStream/gophonic/chat"
)

// Tools are a conversation's, not a voice agent's: see chat.Tool and
// chat.Func. These names remain for programs written against the earlier
// package.

// Tool is chat.Tool.
type Tool = chat.Tool

// Func is chat.Func.
func Func[Args any](name, description string, run func(ctx context.Context, args Args) (string, error)) Tool {
	return chat.Func(name, description, run)
}
