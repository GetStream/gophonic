// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package mcp

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/thesyncim/vibejson"
)

// TestMain runs the fake server when the test binary is started as one.
func TestMain(m *testing.M) {
	if os.Getenv("GOPHONIC_MCP_TEST_SERVER") == "1" {
		serveFake(os.Stdin, os.Stdout, nil)
		return
	}
	os.Exit(m.Run())
}

// fakeLog records what the fake server was told.
type fakeLog struct {
	mu          sync.Mutex
	initialized bool
	cancelled   []int64
	pinged      bool
	added       []string // the arguments add was called with
}

// serveFake is an MCP server with three tools on two pages: add sums a and
// b, fail fails, and slow answers only if it is not cancelled first. It
// pings the client before it introduces itself.
func serveFake(r io.Reader, w io.Writer, log *fakeLog) {
	if log == nil {
		log = &fakeLog{}
	}
	var wmu sync.Mutex
	send := func(s string) {
		wmu.Lock()
		fmt.Fprintln(w, s)
		wmu.Unlock()
	}
	in := bufio.NewScanner(r)
	for in.Scan() {
		line := append([]byte(nil), in.Bytes()...)
		id, hasID, _ := vibejson.GetRaw(line, "/id")
		method := field(line, "/method")
		switch method {
		case "initialize":
			send(`{"jsonrpc":"2.0","id":"s1","method":"ping"}`)
			send(`{"jsonrpc":"2.0","id":` + string(id.Src) + `,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1.2"}}}`)
		case "notifications/initialized":
			log.mu.Lock()
			log.initialized = true
			log.mu.Unlock()
		case "notifications/cancelled":
			v, _, _ := vibejson.GetRaw(line, "/params/requestId")
			n, _ := v.Int64()
			log.mu.Lock()
			log.cancelled = append(log.cancelled, n)
			log.mu.Unlock()
		case "tools/list":
			if field(line, "/params/cursor") == "" {
				send(`{"jsonrpc":"2.0","id":` + string(id.Src) + `,"result":{"tools":[` +
					`{"name":"add","description":"Adds two numbers.","inputSchema":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}},` +
					`{"name":"fail","description":"Fails.","inputSchema":{"type":"object"}}],"nextCursor":"p2"}}`)
			} else {
				send(`{"jsonrpc":"2.0","id":` + string(id.Src) + `,"result":{"tools":[{"name":"slow","inputSchema":{"type":"object"}}]}}`)
			}
		case "tools/call":
			switch field(line, "/params/name") {
			case "add":
				args, _, _ := vibejson.GetRaw(line, "/params/arguments")
				log.mu.Lock()
				log.added = append(log.added, string(args.Src))
				log.mu.Unlock()
				a, _, _ := vibejson.GetRaw(line, "/params/arguments/a")
				b, _, _ := vibejson.GetRaw(line, "/params/arguments/b")
				x, _ := a.Float64()
				y, _ := b.Float64()
				go func(id string) { // answers may come out of order
					time.Sleep(time.Millisecond)
					send(`{"jsonrpc":"2.0","id":` + id + `,"result":{"content":[{"type":"text","text":"` + strconv.FormatFloat(x+y, 'g', -1, 64) + `"}]}}`)
				}(string(id.Src))
			case "fail":
				send(`{"jsonrpc":"2.0","id":` + string(id.Src) + `,"result":{"content":[{"type":"text","text":"no such file"}],"isError":true}}`)
			case "slow":
				// No answer: the caller gives up.
			default:
				send(`{"jsonrpc":"2.0","id":` + string(id.Src) + `,"error":{"code":-32602,"message":"unknown tool"}}`)
			}
		default:
			if hasID && method == "" { // the client's answer to the ping
				log.mu.Lock()
				log.pinged = field(line, "/id") == "s1"
				log.mu.Unlock()
			}
		}
	}
}

func TestClient(t *testing.T) {
	toServer, fromClient := io.Pipe()
	toClient, fromServer := io.Pipe()
	log := &fakeLog{}
	go serveFake(toServer, fromServer, log)
	ctx := context.Background()
	c, err := Connect(ctx, toClient, fromClient)
	if err != nil {
		t.Fatal(err)
	}
	if c.Server.Name != "fake" || c.Server.Version != "1.2" || c.Server.Protocol != "2025-06-18" {
		t.Fatalf("server %+v", c.Server)
	}
	tools, err := c.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 3 || tools[0].Spec().Name != "add" || tools[0].Spec().Description != "Adds two numbers." ||
		tools[0].Spec().Parameters != `{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}` || tools[2].Spec().Name != "slow" {
		t.Fatalf("tools %+v", chat.Specs(tools))
	}
	// Calls in parallel come back to their callers.
	var wg sync.WaitGroup
	var wrong atomic.Int32
	for i := range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := tools[0].Call(ctx, []byte(fmt.Sprintf(`{"a":%d,"b":1}`, i)))
			if err != nil || got != strconv.Itoa(i+1) {
				wrong.Add(1)
			}
		}()
	}
	wg.Wait()
	if wrong.Load() != 0 {
		t.Fatalf("%d of 20 parallel calls came back wrong", wrong.Load())
	}
	if _, err := tools[1].Call(ctx, nil); err == nil || err.Error() != "fail: no such file" {
		t.Fatalf("a failing tool: %v", err)
	}
	// A call given up tells the server.
	short, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	if _, err := tools[2].Call(short, nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("a call given up: %v", err)
	}
	// The tools keep working after it, and the model's calls run through
	// chat.Run as any tool's do.
	if got := chat.Run(ctx, tools, chat.Call{Name: "add", Arguments: []byte(`{"a":2,"b":3}`)}); got != "5" {
		t.Fatalf("chat.Run: %q", got)
	}
	time.Sleep(20 * time.Millisecond)
	log.mu.Lock()
	defer log.mu.Unlock()
	if !log.initialized || !log.pinged || len(log.cancelled) != 1 || len(log.added) != 21 {
		t.Fatalf("server saw initialized %v, ping answered %v, cancellations %v, %d additions", log.initialized, log.pinged, log.cancelled, len(log.added))
	}
	// A server that goes away fails what waits on it.
	fromServer.Close()
	if _, err := tools[0].Call(ctx, []byte(`{"a":1,"b":1}`)); err == nil {
		t.Fatal("a call to a closed server succeeded")
	}
}

// Start runs a server process and ends it on Close.
func TestStart(t *testing.T) {
	t.Setenv("GOPHONIC_MCP_TEST_SERVER", "1")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c, err := Start(ctx, exec.Command(os.Args[0]))
	if err != nil {
		t.Fatal(err)
	}
	tools, err := c.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := tools[0].Call(ctx, []byte(`{"a":20,"b":22}`)); err != nil || got != "42" {
		t.Fatalf("add: %q, %v", got, err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
