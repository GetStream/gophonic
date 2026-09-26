// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

// chatChannel posts to and watches one Stream Chat channel as the token's
// user, through the chat REST API and its realtime websocket. Watching
// gives Gopher the call's text chat as silent context: what people type is
// added to its conversation without being read aloud or answered.

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
	"github.com/thesyncim/vibejson"
)

// restBase and wsBase are Stream Chat's API bases; tests point them at a
// fake server.
var (
	restBase = "https://chat.stream-io-api.com"
	wsBase   = "wss://chat.stream-io-api.com/connect"
)

type chatChannel struct {
	apiKey, token, path string
	edits               bool // the user may edit its own messages here
}

// open creates the channel if nobody has opened the call's chat yet, and
// learns from the channel's capabilities whether c's user may edit its own
// messages: a channel type may let users post but not edit, as Pronto's
// videocall does.
func (c *chatChannel) open() error {
	data, err := c.post("/query", map[string]any{"state": false}, "")
	if err != nil {
		return err
	}
	var resp struct {
		Channel struct {
			OwnCapabilities []string `json:"own_capabilities"`
		} `json:"channel"`
	}
	if err := vibejson.Unmarshal(data, &resp); err != nil {
		return err
	}
	c.edits = slices.Contains(resp.Channel.OwnCapabilities, "update-own-message")
	return nil
}

// send posts text as a message from c's user.
func (c *chatChannel) send(text string) error {
	_, err := c.say(text)
	return err
}

// say posts text as a message from c's user and returns its ID.
func (c *chatChannel) say(text string) (string, error) {
	data, err := c.post("/message", map[string]any{"message": map[string]any{"text": text}}, "")
	if err != nil {
		return "", err
	}
	var resp struct {
		Message wsMessage `json:"message"`
	}
	if err := vibejson.Unmarshal(data, &resp); err != nil {
		return "", err
	}
	return resp.Message.ID, nil
}

// edit replaces the text of c's user's message id.
func (c *chatChannel) edit(id, text string) error {
	_, err := c.request(restBase+"/messages/"+url.PathEscape(id), map[string]any{"message": map[string]any{"text": text}}, "")
	return err
}

// post sends body as JSON to endpoint and returns the response body.
// connectionID, when not empty, ties the request to a websocket connection
// that is watching the channel.
func (c *chatChannel) post(endpoint string, body any, connectionID string) ([]byte, error) {
	return c.request(restBase+c.path+endpoint, body, connectionID)
}

// request posts body as JSON to the API URL u and returns the response body.
func (c *chatChannel) request(u string, body any, connectionID string) ([]byte, error) {
	payload, err := vibejson.Marshal(&body)
	if err != nil {
		return nil, err
	}
	u += "?api_key=" + url.QueryEscape(c.apiKey)
	if connectionID != "" {
		u += "&connection_id=" + url.QueryEscape(connectionID)
	}
	req, err := http.NewRequest(http.MethodPost, u, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", c.token)
	req.Header.Set("Stream-Auth-Type", "jwt")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: %s", resp.Status, data)
	}
	return data, nil
}

// historyLimit caps how much of the channel's past Gopher loads on join.
const historyLimit = 300 // the API's most; Gopher's own captions fill much of it

// healthEvery is how often Gopher pings the realtime connection to keep it
// open; Stream expects one from the client at least every 30 s.
const healthEvery = 25 * time.Second

// wsUser is a chat message's author.
type wsUser struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// wsMessage is a chat message as the query and realtime APIs describe it.
type wsMessage struct {
	ID        string `json:"id"`
	Text      string `json:"text"`
	CreatedAt string `json:"created_at"`
	User      wsUser `json:"user"`
}

// wsEvent is the subset of Stream Chat's realtime events Gopher reads: the
// connect handshake's connection_id, and each new message.
type wsEvent struct {
	Type         string     `json:"type"`
	ConnectionID string     `json:"connection_id"`
	Message      *wsMessage `json:"message"`
}

// watchQuery starts watching a channel and asks for its recent messages.
type watchQuery struct {
	Watch    bool `json:"watch"`
	State    bool `json:"state"`
	Messages struct {
		Limit int `json:"limit"`
	} `json:"messages"`
}

// wsConnectParams identifies Gopher on the connect URL's json parameter.
type wsConnectParams struct {
	UserID      string `json:"user_id"`
	UserDetails struct {
		ID string `json:"id"`
	} `json:"user_details"`
	ServerDeterminesConnectionID bool `json:"server_determines_connection_id"`
}

// healthPing is the client-to-server keepalive frame.
type healthPing struct {
	Type     string `json:"type"`
	ClientID string `json:"client_id"`
}

// wsConn is one realtime connection. Reads go through br, which may already
// hold bytes the websocket handshake read from conn; writes are serialized
// because the health check ticker and the websocket library's own
// control-frame replies can both write to conn.
type wsConn struct {
	br   *bufio.Reader
	conn net.Conn
	mu   sync.Mutex
}

func (w *wsConn) Read(p []byte) (int, error) { return w.br.Read(p) }

func (w *wsConn) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.conn.Write(p)
}

func (w *wsConn) Close() error { return w.conn.Close() }

// connect dials Stream Chat's realtime API and returns the connection once
// its first event has handed back a connection_id.
func (c *chatChannel) connect(ctx context.Context) (*wsConn, string, error) {
	params := wsConnectParams{UserID: self, ServerDeterminesConnectionID: true}
	params.UserDetails.ID = self
	j, err := vibejson.Marshal(&params)
	if err != nil {
		return nil, "", err
	}
	dialURL := wsBase + "?json=" + url.QueryEscape(string(j)) +
		"&api_key=" + url.QueryEscape(c.apiKey) + "&authorization=" + url.QueryEscape(c.token) + "&stream-auth-type=jwt"
	raw, br, _, err := ws.Dial(ctx, dialURL)
	if err != nil {
		return nil, "", err
	}
	if br == nil {
		br = bufio.NewReader(raw)
	}
	conn := &wsConn{br: br, conn: raw}
	data, err := wsutil.ReadServerText(conn)
	if err != nil {
		conn.Close()
		return nil, "", err
	}
	var ev wsEvent
	if err := vibejson.Unmarshal(data, &ev); err != nil {
		conn.Close()
		return nil, "", err
	}
	if ev.ConnectionID == "" {
		conn.Close()
		return nil, "", fmt.Errorf("chat: connect: no connection_id in the first event")
	}
	return conn, ev.ConnectionID, nil
}

// history starts watching connectionID's channel and returns its most
// recent messages, oldest first.
func (c *chatChannel) history(connectionID string) ([]wsMessage, error) {
	q := watchQuery{Watch: true, State: true}
	q.Messages.Limit = historyLimit
	data, err := c.post("/query", q, connectionID)
	if err != nil {
		return nil, err
	}
	var resp struct {
		Messages []wsMessage `json:"messages"`
	}
	if err := vibejson.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	sort.Slice(resp.Messages, func(i, j int) bool { return resp.Messages[i].CreatedAt < resp.Messages[j].CreatedAt })
	return resp.Messages, nil
}

// dial connects and starts watching the channel, retrying with backoff
// until it succeeds or ctx ends.
func (c *chatChannel) dial(ctx context.Context) (conn *wsConn, connectionID string, history []wsMessage, err error) {
	backoff := time.Second
	for {
		if conn, connectionID, err = c.connect(ctx); err == nil {
			if history, err = c.history(connectionID); err == nil {
				return conn, connectionID, history, nil
			}
			conn.Close()
		}
		if ctx.Err() != nil {
			return nil, "", nil, ctx.Err()
		}
		log.Printf("chat: %v; retrying in %v", err, backoff)
		select {
		case <-ctx.Done():
			return nil, "", nil, ctx.Err()
		case <-time.After(backoff):
		}
		if backoff < 30*time.Second {
			backoff *= 2
		}
	}
}

// watch delivers the channel's chat as silent context: onMessage runs for
// its recent history in chronological order, then for each new message as
// it arrives, until ctx ends. It reconnects with backoff if the connection
// drops, without replaying a message onMessage already saw.
func (c *chatChannel) watch(ctx context.Context, onMessage func(userID, name, text string)) error {
	conn, connectionID, history, err := c.dial(ctx)
	if err != nil {
		return err
	}
	seen := make(map[string]bool, historyLimit)
	for _, m := range history {
		if m.ID != "" {
			seen[m.ID] = true
		}
		onMessage(m.User.ID, m.User.Name, m.Text)
	}
	go c.receive(ctx, conn, connectionID, seen, onMessage)
	return nil
}

// receive relays events from conn until it fails, then reconnects with
// backoff and keeps going, until ctx ends.
func (c *chatChannel) receive(ctx context.Context, conn *wsConn, connectionID string, seen map[string]bool, onMessage func(userID, name, text string)) {
	for {
		err := c.pump(ctx, conn, connectionID, seen, onMessage)
		conn.Close()
		if ctx.Err() != nil {
			return
		}
		log.Printf("chat: %v; reconnecting", err)
		var history []wsMessage
		if conn, connectionID, history, err = c.dial(ctx); err != nil {
			return // ctx ended while retrying
		}
		for _, m := range history {
			if m.ID == "" || seen[m.ID] {
				continue
			}
			seen[m.ID] = true
			onMessage(m.User.ID, m.User.Name, m.Text)
		}
	}
}

// pump relays events from conn until it fails or ctx ends: message.new
// events reach onMessage, skipping any already in seen, and a health check
// keeps the socket alive.
func (c *chatChannel) pump(ctx context.Context, conn *wsConn, connectionID string, seen map[string]bool, onMessage func(userID, name, text string)) error {
	ping, err := vibejson.Marshal(&[]healthPing{{Type: "health.check", ClientID: connectionID}})
	if err != nil {
		return err
	}
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			conn.Close() // unblocks the read below
		case <-done:
		}
	}()
	go func() {
		t := time.NewTicker(healthEvery)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				if wsutil.WriteClientText(conn, ping) != nil {
					return
				}
			}
		}
	}()
	for {
		data, err := wsutil.ReadServerText(conn)
		if err != nil {
			return err
		}
		var ev wsEvent
		if err := vibejson.Unmarshal(data, &ev); err != nil || ev.Type != "message.new" || ev.Message == nil {
			continue
		}
		m := ev.Message
		if m.ID != "" {
			if seen[m.ID] {
				continue
			}
			seen[m.ID] = true
		}
		onMessage(m.User.ID, strings.TrimSpace(m.User.Name), m.Text)
	}
}

// liveCaptions writes the call's captions into its chat when closed
// captions are off: what people say as messages, and each of Gopher's
// answers as one message that grows as the voice speaks it, where the
// channel lets Gopher edit its messages; where it does not, each sentence
// is posted once it is spoken. Writes go out in order; while one is under
// way, only the latest text of an answer waits to follow.
type liveCaptions struct {
	room *chatChannel
	mu   sync.Mutex
	wake chan struct{}
	said []string  // messages to post, in order
	next string    // the answer's newest text, when it changed
	done bool      // the answer is final
	cut  sentences // the answer's sentences posted, without edits
}

func newLiveCaptions(room *chatChannel) *liveCaptions {
	l := &liveCaptions{room: room, wake: make(chan struct{}, 1)}
	go l.run()
	return l
}

// say posts a message.
func (l *liveCaptions) say(text string) {
	l.mu.Lock()
	l.said = append(l.said, text)
	l.mu.Unlock()
	l.signal()
}

// answer shows the text of Gopher's answer spoken so far; final ends it.
func (l *liveCaptions) answer(text string, final bool) {
	l.mu.Lock()
	if l.room.edits {
		l.next, l.done = "Gopher: "+text, l.done || final
	} else if spoken := l.cut.next(text, final); spoken != "" {
		l.said = append(l.said, "Gopher: "+spoken)
	}
	l.mu.Unlock()
	l.signal()
}

func (l *liveCaptions) signal() {
	select {
	case l.wake <- struct{}{}:
	default:
	}
}

func (l *liveCaptions) run() {
	id := "" // the answer's message
	for range l.wake {
		for {
			l.mu.Lock()
			said, next, done := l.said, l.next, l.done
			l.said, l.next = nil, ""
			if done && next == "" {
				l.done = false
			}
			l.mu.Unlock()
			if len(said) == 0 && next == "" {
				if done {
					id = ""
				}
				break
			}
			for _, text := range said {
				if _, err := l.room.say(text); err != nil {
					log.Printf("chat: %v", err)
				}
			}
			if next == "" {
				continue
			}
			var err error
			if id == "" {
				id, err = l.room.say(next)
			} else {
				err = l.room.edit(id, next)
			}
			if err != nil {
				log.Printf("chat: %v", err)
			}
		}
	}
}
