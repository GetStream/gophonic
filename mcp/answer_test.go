// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package mcp

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/qwen3"
)

// A model answers with a server's tool as with any other: Qwen3-8B adds two
// numbers by calling the server, and answers with its result.
func TestAnswerOfficial(t *testing.T) {
	m, err := qwen3.Open(testmodels.Path(t, testmodels.Qwen3), qwen3.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	toServer, fromClient := io.Pipe()
	toClient, fromServer := io.Pipe()
	defer fromServer.Close()
	log := &fakeLog{}
	go serveFake(toServer, fromServer, log)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	server, err := Connect(ctx, toClient, fromClient)
	if err != nil {
		t.Fatal(err)
	}
	tools, err := server.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.NewSession("You are a precise assistant. Use the tools you are given.", chat.Specs(tools)...)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Add(chat.User, "What is 48213.7 plus 9127.45? Use the add tool.")
	var b strings.Builder
	if err := chat.Answer(ctx, s, tools, chat.Options{MaxTokens: 512}, &b, 3); err != nil {
		t.Fatal(err)
	}
	log.mu.Lock()
	added := log.added
	log.mu.Unlock()
	if len(added) != 1 || !strings.Contains(b.String(), "57341.15") {
		t.Fatalf("the server added %q; the model answered %q", added, b.String())
	}
	t.Logf("add(%s); %q", added[0], b.String())
}
