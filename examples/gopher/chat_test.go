// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
	"github.com/thesyncim/vibejson"
)

// withBases points restBase and wsBase at a fake server for the duration of
// a test.
func withBases(t *testing.T, rest, wsURL string) {
	t.Helper()
	oldRest, oldWS := restBase, wsBase
	restBase, wsBase = rest, wsURL
	t.Cleanup(func() { restBase, wsBase = oldRest, oldWS })
}

func TestWSEventDecode(t *testing.T) {
	var ev wsEvent
	if err := vibejson.Unmarshal([]byte(`{"type":"health.check","connection_id":"conn-1"}`), &ev); err != nil {
		t.Fatalf("decode health check: %v", err)
	}
	if ev.ConnectionID != "conn-1" || ev.Message != nil {
		t.Fatalf("health check: got %+v", ev)
	}

	ev = wsEvent{}
	raw := `{"type":"message.new","cid":"videocall:x","message":{"id":"m1","text":"hi","created_at":"2026-01-01T00:00:00Z","user":{"id":"alice","name":"Alice"}}}`
	if err := vibejson.Unmarshal([]byte(raw), &ev); err != nil {
		t.Fatalf("decode message.new: %v", err)
	}
	if ev.Type != "message.new" || ev.Message == nil {
		t.Fatalf("message.new: got %+v", ev)
	}
	if ev.Message.Text != "hi" || ev.Message.User.ID != "alice" || ev.Message.User.Name != "Alice" {
		t.Fatalf("message.new fields: got %+v", ev.Message)
	}
}

func TestChatChannelHistory(t *testing.T) {
	var gotQuery string
	var gotBody watchQuery
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		if err := vibejson.Unmarshal(mustBody(t, r), &gotBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"messages":[
			{"id":"m2","text":"second","created_at":"2026-01-01T00:00:02Z","user":{"id":"bob","name":"Bob"}},
			{"id":"m1","text":"first","created_at":"2026-01-01T00:00:01Z","user":{"id":"alice","name":"Alice"}}
		]}`))
	}))
	defer server.Close()
	withBases(t, server.URL, "")

	c := &chatChannel{apiKey: "key", token: "tok", path: "/channels/videocall/test"}
	msgs, err := c.history("conn-42")
	if err != nil {
		t.Fatalf("history: %v", err)
	}
	if !gotBody.Watch || !gotBody.State || gotBody.Messages.Limit != historyLimit {
		t.Fatalf("request body: got %+v", gotBody)
	}
	if gotQuery != "api_key=key&connection_id=conn-42" {
		t.Fatalf("query string: got %q", gotQuery)
	}
	if len(msgs) != 2 || msgs[0].ID != "m1" || msgs[1].ID != "m2" {
		t.Fatalf("expected history sorted oldest first, got %+v", msgs)
	}
}

func mustBody(t *testing.T, r *http.Request) []byte {
	t.Helper()
	data, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	return data
}

// TestChatChannelWatch exercises the whole join flow against a fake Stream
// Chat: the websocket handshake, its first event handing back a
// connection_id, the watch query returning history, and a live message.new
// event arriving afterward.
func TestChatChannelWatch(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/connect", func(w http.ResponseWriter, r *http.Request) {
		conn, _, _, err := ws.UpgradeHTTP(r, w)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer conn.Close()
		if err := wsutil.WriteServerText(conn, []byte(`{"type":"health.check","connection_id":"conn-live"}`)); err != nil {
			t.Errorf("write health check: %v", err)
			return
		}
		// A message from a human, and one gopher would see from its own
		// caption fallback; the test only checks what watch delivers, not
		// the self-filtering main.go does with it.
		msg := `{"type":"message.new","message":{"id":"m-live","text":"hello from alice","user":{"id":"alice","name":"Alice"}}}`
		if err := wsutil.WriteServerText(conn, []byte(msg)); err != nil {
			t.Errorf("write message.new: %v", err)
			return
		}
		// Keep the connection open long enough for the client to read it.
		time.Sleep(300 * time.Millisecond)
	})
	mux.HandleFunc("/channels/videocall/test/query", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"messages":[{"id":"m-hist","text":"earlier","created_at":"2026-01-01T00:00:00Z","user":{"id":"carol","name":"Carol"}}]}`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	withBases(t, server.URL, "ws://"+server.Listener.Addr().String()+"/connect")

	c := &chatChannel{apiKey: "key", token: "tok", path: "/channels/videocall/test"}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	type delivery struct{ userID, name, text string }
	var (
		mu  sync.Mutex
		got []delivery
	)
	live := make(chan struct{})
	if err := c.watch(ctx, func(userID, name, text string) {
		mu.Lock()
		got = append(got, delivery{userID, name, text})
		n := len(got)
		mu.Unlock()
		if n == 2 {
			close(live)
		}
	}); err != nil {
		t.Fatalf("watch: %v", err)
	}

	select {
	case <-live:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for history and the live message")
	}

	mu.Lock()
	defer mu.Unlock()
	if got[0] != (delivery{"carol", "Carol", "earlier"}) {
		t.Fatalf("history delivery: got %+v", got[0])
	}
	if got[1] != (delivery{"alice", "Alice", "hello from alice"}) {
		t.Fatalf("live delivery: got %+v", got[1])
	}
}

// Captions in the chat: what people say is posted; an answer is one
// message, edited as it grows.
func TestLiveCaptions(t *testing.T) {
	var mu sync.Mutex
	var log []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Message struct {
				Text string `json:"text"`
			} `json:"message"`
		}
		if err := vibejson.Unmarshal(mustBody(t, r), &body); err != nil {
			t.Errorf("decode: %v", err)
		}
		mu.Lock()
		log = append(log, r.URL.Path+" "+body.Message.Text)
		mu.Unlock()
		w.Write([]byte(`{"message":{"id":"m1"}}`))
	}))
	defer server.Close()
	withBases(t, server.URL, "")
	l := newLiveCaptions(&chatChannel{apiKey: "key", token: "tok", path: "/channels/videocall/test"})
	l.say("Ana: tell me a joke")
	time.Sleep(50 * time.Millisecond)
	for _, s := range []string{"Gopher: Why", "Gopher: Why did the", "Gopher: Why did the chicken cross?"} {
		l.answer(s, s == "Gopher: Why did the chicken cross?")
		time.Sleep(50 * time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	want := []string{"/channels/videocall/test/message Ana: tell me a joke", "/channels/videocall/test/message Gopher: Why",
		"/messages/m1 Gopher: Why did the", "/messages/m1 Gopher: Why did the chicken cross?"}
	if strings.Join(log, "|") != strings.Join(want, "|") {
		t.Fatalf("chat\n%q\nwant\n%q", log, want)
	}
}
