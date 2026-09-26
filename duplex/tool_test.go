// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package duplex

import (
	"context"
	"testing"
)

// The names kept for earlier programs are chat's.
func TestFuncIsChatFunc(t *testing.T) {
	var tool Tool = Func("echo", "Echoes.", func(_ context.Context, args struct {
		Text string `json:"text"`
	}) (string, error) {
		return args.Text, nil
	})
	got, err := tool.Call(context.Background(), []byte(`{"text": "hi"}`))
	if err != nil || got != "hi" || tool.Spec().Name != "echo" {
		t.Fatalf("%q %v %+v", got, err, tool.Spec())
	}
}
