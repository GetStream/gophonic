// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package httpserver

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/internal/transcriptformat"
	"github.com/thesyncim/vibejson"
	"github.com/thesyncim/vibejson/document"
)

// maxJSONBody bounds a JSON request body: a conversation with its tools.
const maxJSONBody = 4 << 20

// chatRequest is what gophonic serves of an OpenAI chat completion request.
type chatRequest struct {
	model    string
	messages []chatMessage
	tools    []chat.ToolSpec
	stream   bool
	opts     chat.Options
}

// chatMessage is one message of a conversation: an assistant message may
// hold the tool calls it made.
type chatMessage struct {
	role  chat.Role
	text  string
	calls []chat.Call
}

var errChatField = errors.New("unsupported")

// parseChat reads an OpenAI chat completion request. Unknown fields are
// ignored, as OpenAI's own servers do; fields gophonic cannot honor (n > 1,
// stop, content other than text) fail.
func parseChat(body []byte, req *chatRequest) error {
	*req = chatRequest{messages: req.messages[:0], tools: req.tools[:0]}
	// OpenAI's defaults: sampling at temperature 1 over every token.
	req.opts.Temperature, req.opts.TopP = 1, 1
	return vibejson.EachObject(body, func(key string, v vibejson.RawValue) error {
		switch key {
		case "model":
			req.model = text(v)
		case "messages":
			return vibejson.EachArray(v.Src, func(_ int, m vibejson.RawValue) error {
				msg, err := parseMessage(m)
				req.messages = append(req.messages, msg)
				return err
			})
		case "tools":
			return vibejson.EachArray(v.Src, func(_ int, t vibejson.RawValue) error {
				fn, ok, err := t.Pointer("/function")
				if err != nil || !ok {
					return fmt.Errorf("a tool must be a function: %w", errChatField)
				}
				spec := chat.ToolSpec{Name: field(fn, "/name"), Description: field(fn, "/description")}
				if params, ok, _ := fn.Pointer("/parameters"); ok {
					spec.Parameters = string(params.Src)
				}
				req.tools = append(req.tools, spec)
				return nil
			})
		case "stream":
			req.stream, _ = v.Bool()
		case "max_tokens", "max_completion_tokens":
			if n, ok := v.Int64(); ok {
				req.opts.MaxTokens = int(n)
			}
		case "temperature":
			if f, ok := v.Float64(); ok {
				req.opts.Temperature = float32(f)
			}
		case "top_p":
			if f, ok := v.Float64(); ok {
				req.opts.TopP = float32(f)
			}
		case "top_k":
			if n, ok := v.Int64(); ok {
				req.opts.TopK = int(n)
			}
		case "presence_penalty":
			if f, ok := v.Float64(); ok {
				req.opts.Presence = float32(f)
			}
		case "seed":
			if n, ok := v.Uint64(); ok {
				req.opts.Seed = n
			}
		case "n":
			if n, ok := v.Int64(); ok && n != 1 {
				return fmt.Errorf("n %d: one choice per request: %w", n, errChatField)
			}
		case "stop":
			if !v.IsNull() {
				return fmt.Errorf("stop sequences: %w", errChatField)
			}
		}
		return nil
	})
}

// parseMessage reads one message: its role, its text (a string or text
// parts), and an assistant's tool calls.
func parseMessage(m vibejson.RawValue) (chatMessage, error) {
	var msg chatMessage
	err := vibejson.EachObject(m.Src, func(key string, v vibejson.RawValue) error {
		switch key {
		case "role":
			switch role := text(v); role {
			case "system", "developer":
				msg.role = chat.System
			case "user":
				msg.role = chat.User
			case "assistant":
				msg.role = chat.Assistant
			case "tool":
				msg.role = chat.ToolResult
			default:
				return fmt.Errorf("role %q: %w", role, errChatField)
			}
		case "content":
			switch v.Kind() {
			case document.String:
				msg.text = text(v)
			case document.Array:
				return vibejson.EachArray(v.Src, func(_ int, part vibejson.RawValue) error {
					if kind := field(part, "/type"); kind != "text" {
						return fmt.Errorf("content of type %q: %w", kind, errChatField)
					}
					msg.text += field(part, "/text")
					return nil
				})
			}
		case "tool_calls":
			return vibejson.EachArray(v.Src, func(_ int, c vibejson.RawValue) error {
				fn, ok, err := c.Pointer("/function")
				if err != nil || !ok {
					return fmt.Errorf("a tool call must call a function: %w", errChatField)
				}
				msg.calls = append(msg.calls, chat.Call{Name: field(fn, "/name"), Arguments: []byte(field(fn, "/arguments"))})
				return nil
			})
		}
		return nil
	})
	return msg, err
}

// text is a string value's text, or "".
func text(v vibejson.RawValue) string {
	s, _, _ := v.Text()
	return s
}

// field is the text of the string at pointer in v, or "".
func field(v vibejson.RawValue, pointer string) string {
	f, ok, _ := v.Pointer(pointer)
	if !ok {
		return ""
	}
	return text(f)
}

// completions counts chat completions, for their ids.
var completions atomic.Uint64

// chatCompletions answers POST /v1/chat/completions as OpenAI's API does,
// with a chat.Generator lane: the request's conversation is replayed into a
// new session, and the reply comes back whole or, with "stream": true, as
// server-sent events while it is written. The server runs no tools: the
// calls a reply makes come back as tool_calls, for the client to run and
// answer in its next request.
func (s *Server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	l, ok := s.admit(w, r)
	if !ok {
		return
	}
	defer s.done(l)
	nbody, err := readBodyInto(r.Body, l.upload[:maxJSONBody])
	if err != nil {
		writeError(w, http.StatusBadRequest, "could not read a JSON body of at most 4 MiB")
		return
	}
	req := &l.chat
	if err := parseChat(l.upload[:nbody], req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid chat request: "+err.Error())
		return
	}
	if len(req.messages) == 0 {
		writeError(w, http.StatusBadRequest, "a chat request needs messages")
		return
	}
	lease, err := acquire[chat.Generator](s, unsafe.Slice(unsafe.StringData(req.model), len(req.model)))
	if err != nil {
		writeLeaseError(w, err)
		return
	}
	defer lease.Release()
	system, messages := "", req.messages
	if messages[0].role == chat.System {
		system, messages = messages[0].text, messages[1:]
	}
	session, err := lease.Lane.NewSession(system, req.tools...)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	defer session.Close()
	for _, m := range messages {
		if len(m.calls) > 0 {
			err = session.AddCalls(m.text, m.calls)
		} else {
			err = session.Add(m.role, m.text)
		}
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}
	// What every chunk and the whole reply share.
	id := completions.Add(1)
	head := append(l.response[:0], `{"id":"chatcmpl-`...)
	head = strconv.AppendUint(head, id, 36)
	head = append(head, `","object":"chat.completion`...)
	if req.stream {
		head = append(head, ".chunk"...)
	}
	head = append(head, `","created":`...)
	head = strconv.AppendInt(head, time.Now().Unix(), 10)
	head = append(head, `,"model":`...)
	head = transcriptformat.AppendJSONString(head, s.modelName(lease.Model().Path()))
	head = append(head, `,"choices":[{"index":0,`...)
	if req.stream {
		s.streamChat(w, r, l, head, id, session, req.opts)
		return
	}
	l.said = l.said[:0]
	if err := session.Reply(r.Context(), req.opts, (*appender)(&l.said)); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	calls := session.Calls()
	out := append(head, `"message":{"role":"assistant","content":`...)
	out = transcriptformat.AppendJSONString(out, l.said)
	if len(calls) > 0 {
		out = append(out, `,"tool_calls":`...)
		out = appendCalls(out, id, calls, false)
	}
	out = append(out, `},"finish_reason":`...)
	out = appendFinish(out, calls)
	out = append(out, "}]}\n"...)
	l.response = out
	w.Header()["Content-Type"] = jsonContentType
	_, _ = w.Write(out)
}

// streamChat writes the reply as server-sent events: a chunk for the role,
// one per piece the model writes, one for the tool calls, one to finish,
// and [DONE].
func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, l *slot, head []byte, id uint64, session chat.Session, opts chat.Options) {
	flusher, _ := w.(http.Flusher)
	w.Header()["Content-Type"] = eventStreamContentType
	w.Header()["Cache-Control"] = noCache
	e := &events{w: w, flusher: flusher, head: head, buf: l.said[:0], id: id}
	e.send(`{"role":"assistant","content":""}`, nil, "null")
	err := session.Reply(r.Context(), opts, e)
	if err != nil && r.Context().Err() == nil {
		// The status is sent; the error goes as an event, as OpenAI's do.
		message := err.Error()
		e.buf = append(e.buf[:0], `data: {"error":{"message":`...)
		e.buf = transcriptformat.AppendJSONString(e.buf, unsafe.Slice(unsafe.StringData(message), len(message)))
		e.buf = append(e.buf, "}}\n\n"...)
		_, _ = w.Write(e.buf)
		l.said = e.buf
		return
	}
	if calls := session.Calls(); len(calls) > 0 {
		e.send("", calls, "null")
		e.send("{}", nil, `"tool_calls"`)
	} else {
		e.send("{}", nil, `"stop"`)
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	if flusher != nil {
		flusher.Flush()
	}
	l.said = e.buf
}

// events writes a streamed reply's chunks; it is the io.Writer Reply
// writes the text to.
type events struct {
	w       http.ResponseWriter
	flusher http.Flusher
	head    []byte // the chunk's id, object, created, model, and choices opening
	buf     []byte
	id      uint64
}

func (e *events) Write(piece []byte) (int, error) {
	e.buf = append(append(append(e.buf[:0], "data: "...), e.head...), `"delta":{"content":`...)
	e.buf = transcriptformat.AppendJSONString(e.buf, piece)
	e.buf = append(e.buf, "},\"finish_reason\":null}]}\n\n"...)
	if _, err := e.w.Write(e.buf); err != nil {
		return 0, err
	}
	if e.flusher != nil {
		e.flusher.Flush()
	}
	return len(piece), nil
}

// send writes a chunk whose delta is delta, or the calls, and whose
// finish_reason is finish (JSON).
func (e *events) send(delta string, calls []chat.Call, finish string) {
	e.buf = append(append(append(e.buf[:0], "data: "...), e.head...), `"delta":`...)
	if calls != nil {
		e.buf = append(e.buf, `{"tool_calls":`...)
		e.buf = appendCalls(e.buf, e.id, calls, true)
		e.buf = append(e.buf, '}')
	} else {
		e.buf = append(e.buf, delta...)
	}
	e.buf = append(e.buf, `,"finish_reason":`...)
	e.buf = append(e.buf, finish...)
	e.buf = append(e.buf, "}]}\n\n"...)
	_, _ = e.w.Write(e.buf)
	if e.flusher != nil {
		e.flusher.Flush()
	}
}

// appendCalls appends tool calls as OpenAI lists them, with ids unique to
// the completion; streamed calls carry their index.
func appendCalls(dst []byte, id uint64, calls []chat.Call, indexed bool) []byte {
	dst = append(dst, '[')
	for i, c := range calls {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = append(dst, '{')
		if indexed {
			dst = append(dst, `"index":`...)
			dst = strconv.AppendInt(dst, int64(i), 10)
			dst = append(dst, ',')
		}
		dst = append(dst, `"id":"call_`...)
		dst = strconv.AppendUint(dst, id, 36)
		dst = append(dst, '_')
		dst = strconv.AppendInt(dst, int64(i), 10)
		dst = append(dst, `","type":"function","function":{"name":`...)
		dst = transcriptformat.AppendJSONString(dst, unsafe.Slice(unsafe.StringData(c.Name), len(c.Name)))
		dst = append(dst, `,"arguments":`...)
		dst = transcriptformat.AppendJSONString(dst, c.Arguments)
		dst = append(dst, "}}"...)
	}
	return append(dst, ']')
}

func appendFinish(dst []byte, calls []chat.Call) []byte {
	if len(calls) > 0 {
		return append(dst, `"tool_calls"`...)
	}
	return append(dst, `"stop"`...)
}

// appender is an io.Writer that appends to a byte slice.
type appender []byte

func (a *appender) Write(p []byte) (int, error) {
	*a = append(*a, p...)
	return len(p), nil
}

var (
	eventStreamContentType = []string{"text/event-stream"}
	noCache                = []string{"no-cache"}
)
