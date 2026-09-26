// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package mcp connects to Model Context Protocol servers and offers their
// tools as chat.Tools: a model calls them as it calls a Go function made
// with chat.Func, in chat.Answer, in a duplex agent, or through the
// server's chat endpoint. It speaks JSON-RPC over a server's standard input
// and output, MCP's stdio transport.
//
//	server, err := mcp.Start(ctx, exec.Command("npx", "-y", "@modelcontextprotocol/server-filesystem", dir))
//	defer server.Close()
//	tools, err := server.Tools(ctx)
//	s, err := gen.NewSession("You are helpful.", chat.Specs(tools)...)
//	err = chat.Answer(ctx, s, tools, chat.Options{}, os.Stdout, 4)
package mcp

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/thesyncim/vibejson"
)

// protocolVersion is the MCP revision the client asks for; the server
// answers with the one it speaks.
const protocolVersion = "2025-06-18"

// Client is a session with one MCP server. Its methods may be called
// concurrently: calls to the server are multiplexed on one connection.
type Client struct {
	// Server names the server and the protocol revision it speaks, as it
	// introduced itself.
	Server struct{ Name, Version, Protocol string }

	wmu     sync.Mutex
	w       io.Writer
	next    atomic.Int64
	mu      sync.Mutex
	pending map[int64]chan response
	done    chan struct{} // closed when the server's output ends
	err     error         // why it ended
	stop    func() error  // ends the server Start ran
}

type response struct {
	result []byte
	err    error
}

// Start runs cmd as an MCP server and connects to it over its standard
// input and output; cmd's environment, directory and standard error are the
// caller's to set. ctx bounds the start and the introductions; Close ends
// the server.
func Start(ctx context.Context, cmd *exec.Cmd) (*Client, error) {
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("mcp: start %s: %w", cmd.Path, err)
	}
	stop := func() error {
		// A server ends when its input does; one that lingers is killed.
		stdin.Close()
		exited := make(chan error, 1)
		go func() { exited <- cmd.Wait() }()
		select {
		case err := <-exited:
			return err
		case <-time.After(2 * time.Second):
			cmd.Process.Kill()
			return <-exited
		}
	}
	c, err := Connect(ctx, stdout, stdin)
	if err != nil {
		stop()
		return nil, err
	}
	c.stop = stop
	return c, nil
}

// Connect speaks MCP over r, the server's output, and w, its input, and
// introduces the client, within ctx.
func Connect(ctx context.Context, r io.Reader, w io.Writer) (*Client, error) {
	c := &Client{w: w, pending: map[int64]chan response{}, done: make(chan struct{})}
	go c.read(r)
	res, err := c.call(ctx, "initialize", []byte(`{"protocolVersion":"`+protocolVersion+
		`","capabilities":{},"clientInfo":{"name":"gophonic","version":"1"}}`))
	if err != nil {
		return nil, err
	}
	c.Server.Protocol = field(res, "/protocolVersion")
	c.Server.Name = field(res, "/serverInfo/name")
	c.Server.Version = field(res, "/serverInfo/version")
	if err := c.send([]byte(`{"jsonrpc":"2.0","method":"notifications/initialized"}`)); err != nil {
		return nil, err
	}
	return c, nil
}

// Tools lists the server's tools, every page of them, as chat.Tools that
// call the server.
func (c *Client) Tools(ctx context.Context) ([]chat.Tool, error) {
	var tools []chat.Tool
	params := []byte("{}")
	for {
		res, err := c.call(ctx, "tools/list", params)
		if err != nil {
			return nil, err
		}
		list, ok, _ := vibejson.GetRaw(res, "/tools")
		if ok {
			err = vibejson.EachArray(list.Src, func(_ int, t vibejson.RawValue) error {
				spec := chat.ToolSpec{Name: field(t.Src, "/name"), Description: field(t.Src, "/description")}
				if schema, ok, _ := t.Pointer("/inputSchema"); ok {
					spec.Parameters = string(schema.Src)
				}
				tools = append(tools, &tool{c, spec})
				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("mcp: tools/list: %w", err)
			}
		}
		cursor := field(res, "/nextCursor")
		if cursor == "" {
			return tools, nil
		}
		params = appendString(append([]byte(nil), `{"cursor":`...), cursor)
		params = append(params, '}')
	}
}

// Close ends the server Start ran: its input is closed, and a server that
// has not exited two seconds later is killed. The connection of a client
// made with Connect is the caller's to close.
func (c *Client) Close() error {
	if c.stop != nil {
		return c.stop()
	}
	return nil
}

// tool is a server's tool.
type tool struct {
	c    *Client
	spec chat.ToolSpec
}

func (t *tool) Spec() chat.ToolSpec { return t.spec }

// Call calls the tool with arguments, a JSON object, and returns the text
// of its result. A result the server marks as an error is returned as one,
// for the model to read.
func (t *tool) Call(ctx context.Context, arguments []byte) (string, error) {
	if len(arguments) == 0 {
		arguments = []byte("{}")
	}
	if !vibejson.Valid(arguments) {
		return "", fmt.Errorf("mcp: %s: arguments are not JSON", t.spec.Name)
	}
	params := appendString(append([]byte(nil), `{"name":`...), t.spec.Name)
	params = append(append(append(params, `,"arguments":`...), arguments...), '}')
	res, err := t.c.call(ctx, "tools/call", params)
	if err != nil {
		return "", err
	}
	var text []byte
	if content, ok, _ := vibejson.GetRaw(res, "/content"); ok {
		vibejson.EachArray(content.Src, func(_ int, item vibejson.RawValue) error {
			if len(text) > 0 {
				text = append(text, '\n')
			}
			switch kind := field(item.Src, "/type"); kind {
			case "text":
				text = append(text, field(item.Src, "/text")...)
			case "resource":
				text = append(text, field(item.Src, "/resource/text")...)
			case "resource_link":
				text = append(text, field(item.Src, "/uri")...)
			default:
				text = append(text, "["+kind+"]"...)
			}
			return nil
		})
	}
	if len(text) == 0 {
		// A result with structured content only.
		if structured, ok, _ := vibejson.GetRaw(res, "/structuredContent"); ok {
			text = structured.Src
		}
	}
	if failed, ok, _ := vibejson.GetRaw(res, "/isError"); ok {
		if b, _ := failed.Bool(); b {
			return "", fmt.Errorf("%s: %s", t.spec.Name, text)
		}
	}
	return string(text), nil
}

// call sends a request and waits for its response. When ctx ends first,
// the server is told the request is cancelled.
func (c *Client) call(ctx context.Context, method string, params []byte) ([]byte, error) {
	id := c.next.Add(1)
	ch := make(chan response, 1)
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return nil, c.err
	}
	c.pending[id] = ch
	c.mu.Unlock()
	msg := strconv.AppendInt(append([]byte(nil), `{"jsonrpc":"2.0","id":`...), id, 10)
	msg = append(append(append(msg, `,"method":"`...), method...), `","params":`...)
	msg = append(append(msg, params...), '}')
	if err := c.send(msg); err != nil {
		c.forget(id)
		return nil, err
	}
	select {
	case r := <-ch:
		if r.err != nil {
			return nil, fmt.Errorf("mcp: %s: %w", method, r.err)
		}
		return r.result, nil
	case <-ctx.Done():
		c.forget(id)
		note := strconv.AppendInt(append([]byte(nil), `{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":`...), id, 10)
		c.send(append(note, `,"reason":"the caller gave up"}}`...))
		return nil, ctx.Err()
	}
}

func (c *Client) forget(id int64) {
	c.mu.Lock()
	delete(c.pending, id)
	c.mu.Unlock()
}

// send writes one message: a line of JSON.
func (c *Client) send(msg []byte) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	if _, err := c.w.Write(append(msg, '\n')); err != nil {
		return fmt.Errorf("mcp: write: %w", err)
	}
	return nil
}

// read reads the server's messages: it hands responses to their calls,
// answers the server's requests (ping; anything else is a method the client
// lacks), and ignores notifications. It never waits on a write, which could
// wait on the server's reading while the server waits on its writing.
func (c *Client) read(r io.Reader) {
	in := bufio.NewReaderSize(r, 64<<10)
	var err error
	for {
		var line []byte
		if line, err = in.ReadBytes('\n'); len(line) > 0 {
			c.handle(line)
		}
		if err != nil {
			break
		}
	}
	if errors.Is(err, io.EOF) {
		err = errors.New("mcp: the server closed the connection")
	}
	c.mu.Lock()
	c.err = err
	for id, ch := range c.pending {
		ch <- response{err: err}
		delete(c.pending, id)
	}
	c.mu.Unlock()
	close(c.done)
}

func (c *Client) handle(line []byte) {
	id, hasID, _ := vibejson.GetRaw(line, "/id")
	method := field(line, "/method")
	switch {
	case method != "" && hasID:
		reply := append(append([]byte(nil), `{"jsonrpc":"2.0","id":`...), id.Src...)
		if method == "ping" {
			reply = append(reply, `,"result":{}}`...)
		} else {
			reply = append(reply, `,"error":{"code":-32601,"message":"method not found"}}`...)
		}
		go c.send(reply)
	case hasID:
		n, ok := id.Int64()
		if !ok {
			return
		}
		c.mu.Lock()
		ch := c.pending[n]
		delete(c.pending, n)
		c.mu.Unlock()
		if ch == nil {
			return // a call given up
		}
		if e, ok, _ := vibejson.GetRaw(line, "/error"); ok {
			ch <- response{err: errors.New(field(e.Src, "/message"))}
			return
		}
		res, _, _ := vibejson.GetRaw(line, "/result")
		ch <- response{result: append([]byte(nil), res.Src...)}
	}
}

// field is the text of the string at pointer in src, or "".
func field(src []byte, pointer string) string {
	v, ok, _ := vibejson.GetRaw(src, pointer)
	if !ok {
		return ""
	}
	s, _, _ := v.Text()
	return s
}

// appendString appends s as a JSON string.
func appendString(dst []byte, s string) []byte {
	const hex = "0123456789abcdef"
	dst = append(dst, '"')
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == '"' || c == '\\':
			dst = append(dst, '\\', c)
		case c < 0x20:
			dst = append(dst, '\\', 'u', '0', '0', hex[c>>4], hex[c&15])
		default:
			dst = append(dst, c)
		}
	}
	return append(dst, '"')
}
