// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package httpserver

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/speech"
	"github.com/GetStream/gophonic/speech/speechtest"
	"github.com/thesyncim/vibejson"
)

// fake is a model that classifies audio and text, counting the questions
// it prepares.
type fake struct{ questions *atomic.Int32 }

func (fake) Labels() []string { return []string{"incomplete", "complete"} }

func (fake) ClassifyInto(pcm []float32, _, _ int, probs []float32) error {
	probs[0], probs[1] = 0.25, 0.75
	return nil
}

func (fake) Close() error { return nil }

func (f fake) Classifier(question string, labels []string) (speech.TextClassifier, error) {
	f.questions.Add(1)
	return fakeText{labels}, nil
}

type fakeText struct{ labels []string }

func (t fakeText) Labels() []string { return t.labels }

// ClassifyInto gives the label the text names all the probability.
func (t fakeText) ClassifyInto(_ context.Context, text string, probs []float32) error {
	for i, l := range t.labels {
		probs[i] = 0
		if strings.Contains(text, l) {
			probs[i] = 1
		}
	}
	return nil
}

func (fakeText) Close() error { return nil }

var fakeQuestions atomic.Int32

var registerFake = sync.OnceFunc(func() {
	gophonic.Register(gophonic.Format{
		Name:  "fake",
		Match: func(p string) bool { b, _ := os.ReadFile(p); return string(b) == "FAKEMODL" },
		Open: func(path string, _ gophonic.Options) (*gophonic.Model, error) {
			m := gophonic.NewModel("fake", path, nil)
			gophonic.Provide(m, func() (speech.AudioClassifier, error) { return fake{&fakeQuestions}, nil })
			gophonic.Provide(m, func() (chat.Generator, error) { return fakeChat{}, nil })
			gophonic.Provide(m, func() (speech.Synthesizer, error) { return speechtest.NewTone(10 * time.Millisecond), nil })
			return gophonic.Provide(m, func() (speech.ZeroShot, error) { return fake{&fakeQuestions}, nil }), nil
		},
	})
})

func fakeServer(t *testing.T) *Server {
	registerFake()
	path := filepath.Join(t.TempDir(), "judge.gophonic")
	if err := os.WriteFile(path, []byte("FAKEMODL"), 0o644); err != nil {
		t.Fatal(err)
	}
	return newTestServer(t, 5, path)
}

func serve(s *Server, method, route, contentType string, body []byte) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, route, bytes.NewReader(body))
	if contentType != "" {
		r.Header.Set("Content-Type", contentType)
	}
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	return w
}

func TestModelList(t *testing.T) {
	s := fakeServer(t)
	var list modelList
	w := serve(s, http.MethodGet, "/v1/models", "", nil)
	if err := vibejson.Unmarshal(w.Body.Bytes(), &list); err != nil || w.Code != http.StatusOK {
		t.Fatalf("status %d, %v: %s", w.Code, err, w.Body)
	}
	if list.Object != "list" || len(list.Data) != 1 || list.Data[0].ID != "judge" || list.Data[0].Object != "model" || list.Data[0].Loaded {
		t.Fatalf("model list %+v", list)
	}
}

func TestClassifyAudio(t *testing.T) {
	s := fakeServer(t)
	wav := make([]byte, 44+4*1600)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], uint32(len(wav)-8))
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 3)
	binary.LittleEndian.PutUint16(wav[22:], 1)
	binary.LittleEndian.PutUint32(wav[24:], 16000)
	binary.LittleEndian.PutUint32(wav[28:], 64000)
	binary.LittleEndian.PutUint16(wav[32:], 4)
	binary.LittleEndian.PutUint16(wav[34:], 32)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 4*1600)
	for _, model := range []string{"", "judge"} {
		r := multipartRequest(t, "turn.wav", wav, map[string]string{"model": model})
		r.URL.Path = "/v1/audio/classifications"
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		want := `{"model":"judge","classes":[{"label":"incomplete","probability":0.25},{"label":"complete","probability":0.75}]}` + "\n"
		if w.Code != http.StatusOK || w.Body.String() != want {
			t.Fatalf("model %q: status %d: %s", model, w.Code, w.Body)
		}
	}
	r := multipartRequest(t, "turn.wav", wav, map[string]string{"model": "nope"})
	r.URL.Path = "/v1/audio/classifications"
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Fatalf("unknown model: status %d: %s", w.Code, w.Body)
	}
}

func TestClassifyText(t *testing.T) {
	s := fakeServer(t)
	fakeQuestions.Store(0)
	one := `{"model":"judge","input":"this is spam","question":"Is it spam?","labels":["spam","ham"]}`
	want := `{"model":"judge","results":[{"classes":[{"label":"spam","probability":1},{"label":"ham","probability":0}]}]}` + "\n"
	for range 3 {
		w := serve(s, http.MethodPost, "/v1/classifications", "application/json", []byte(one))
		if w.Code != http.StatusOK || w.Body.String() != want {
			t.Fatalf("status %d: %s", w.Code, w.Body)
		}
	}
	if n := fakeQuestions.Load(); n != 1 {
		t.Fatalf("a repeated question was prepared %d times", n)
	}
	many := `{"input":["ham and eggs","spam"],"question":"Is it spam?","labels":["spam","ham"]}`
	w := serve(s, http.MethodPost, "/v1/classifications", "application/json", []byte(many))
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"results":[{"classes":[{"label":"spam","probability":0},{"label":"ham","probability":1}]},{"classes":[{"label":"spam","probability":1}`) {
		t.Fatalf("batch: status %d: %s", w.Code, w.Body)
	}
	for _, bad := range []string{`{"input":"x","question":"q","labels":["one"]}`, `{"question":"q"}`, `nope`} {
		if w := serve(s, http.MethodPost, "/v1/classifications", "application/json", []byte(bad)); w.Code != http.StatusBadRequest {
			t.Fatalf("%s: status %d: %s", bad, w.Code, w.Body)
		}
	}
	// A model without a text classifier of its own needs a question.
	if w := serve(s, http.MethodPost, "/v1/classifications", "application/json", []byte(`{"input":"x"}`)); w.Code != http.StatusBadRequest {
		t.Fatalf("no question: status %d: %s", w.Code, w.Body)
	}
}

// fakeChat's sessions echo the last message in pieces, or, offered tools
// and asked the time, call now. The last conversation is kept.
type fakeChat struct{}

var lastConversation atomic.Pointer[fakeConversation]

func (fakeChat) NewSession(system string, tools ...chat.ToolSpec) (chat.Session, error) {
	c := &fakeConversation{log: []string{"system:" + system}, tools: tools}
	lastConversation.Store(c)
	return c, nil
}
func (fakeChat) Close() error { return nil }

type fakeConversation struct {
	log   []string
	tools []chat.ToolSpec
	calls []chat.Call
	opts  chat.Options
}

func (c *fakeConversation) Add(role chat.Role, text string) error {
	c.log = append(c.log, fmt.Sprintf("%d:%s", role, text))
	return nil
}

func (c *fakeConversation) AddCalls(text string, calls []chat.Call) error {
	for _, call := range calls {
		text += fmt.Sprintf("[%s %s]", call.Name, call.Arguments)
	}
	c.log = append(c.log, "calls:"+text)
	return nil
}

func (c *fakeConversation) Reply(_ context.Context, opts chat.Options, w io.Writer) error {
	c.opts = opts
	last := c.log[len(c.log)-1]
	_, text, _ := strings.Cut(last, ":")
	if len(c.tools) > 0 && strings.Contains(text, "time") {
		c.calls = []chat.Call{{Name: "now", Arguments: []byte(`{"zone":"UTC"}`)}}
		return nil
	}
	reply := "You said: " + text
	for len(reply) > 0 {
		n := min(3, len(reply))
		if _, err := w.Write([]byte(reply[:n])); err != nil {
			return err
		}
		reply = reply[n:]
	}
	return nil
}

func (c *fakeConversation) Calls() []chat.Call { return c.calls }
func (c *fakeConversation) Finished(context.Context, chat.Role, string) (float32, error) {
	return 1, nil
}
func (c *fakeConversation) Prefill(context.Context) error { return nil }
func (c *fakeConversation) Truncate(int) error            { return nil }
func (c *fakeConversation) Checkpoint() int               { return 0 }
func (c *fakeConversation) Restore(int) error             { return nil }
func (c *fakeConversation) Close() error                  { return nil }

// The chat endpoint replays the conversation, including the calls an
// assistant made and their results, and answers as OpenAI's does.
func TestChatCompletions(t *testing.T) {
	s := fakeServer(t)
	body := `{"model":"judge","messages":[{"role":"system","content":"Be brief."},
		{"role":"user","content":[{"type":"text","text":"What "},{"type":"text","text":"time?"}]},
		{"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"now","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"x","content":"noon"},
		{"role":"user","content":"hi"}],"temperature":0.5,"max_tokens":20,"seed":7}`
	w := serve(s, http.MethodPost, "/v1/chat/completions", "application/json", []byte(body))
	var reply struct {
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string                           `json:"id"`
					Function struct{ Name, Arguments string } `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := vibejson.Unmarshal(w.Body.Bytes(), &reply); err != nil || w.Code != http.StatusOK {
		t.Fatalf("status %d, %v: %s", w.Code, err, w.Body)
	}
	if reply.Object != "chat.completion" || reply.Model != "judge" || reply.Choices[0].Message.Content != "You said: hi" || reply.Choices[0].FinishReason != "stop" {
		t.Fatalf("reply %+v", reply)
	}
	c := lastConversation.Load()
	want := []string{"system:Be brief.", "1:What time?", "calls:[now {}]", "3:noon", "1:hi"}
	if !slices.Equal(c.log[:len(want)], want) || c.opts.Temperature != 0.5 || c.opts.MaxTokens != 20 || c.opts.Seed != 7 {
		t.Fatalf("conversation %q, options %+v", c.log, c.opts)
	}
	// Offered tools and asked the time, the model calls one.
	tools := `{"messages":[{"role":"user","content":"what time is it?"}],"tools":[{"type":"function","function":{"name":"now","description":"The time.","parameters":{"type":"object"}}}]}`
	w = serve(s, http.MethodPost, "/v1/chat/completions", "application/json", []byte(tools))
	if err := vibejson.Unmarshal(w.Body.Bytes(), &reply); err != nil || w.Code != http.StatusOK {
		t.Fatalf("status %d, %v: %s", w.Code, err, w.Body)
	}
	calls := reply.Choices[0].Message.ToolCalls
	if len(calls) != 1 || calls[0].Function.Name != "now" || calls[0].Function.Arguments != `{"zone":"UTC"}` || reply.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("reply %s", w.Body)
	}
	if spec := lastConversation.Load().tools; len(spec) != 1 || spec[0].Parameters != `{"type":"object"}` || spec[0].Description != "The time." {
		t.Fatalf("tools %+v", spec)
	}
	for _, bad := range []string{`{"messages":[]}`, `{"messages":[{"role":"user","content":"x"}],"n":2}`,
		`{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}`, `{"messages":[{"role":"wizard","content":"x"}]}`} {
		if w := serve(s, http.MethodPost, "/v1/chat/completions", "application/json", []byte(bad)); w.Code != http.StatusBadRequest {
			t.Fatalf("%s: status %d: %s", bad, w.Code, w.Body)
		}
	}
}

// Streamed, the reply comes as server-sent events: the role, a chunk per
// piece, the calls, the finish, and [DONE].
func TestChatCompletionsStream(t *testing.T) {
	s := fakeServer(t)
	w := serve(s, http.MethodPost, "/v1/chat/completions", "application/json", []byte(`{"stream":true,"messages":[{"role":"user","content":"hello"}]}`))
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d, %s", w.Code, w.Header().Get("Content-Type"))
	}
	var text, finish string
	events := strings.Split(strings.TrimSuffix(w.Body.String(), "\n\n"), "\n\n")
	if events[len(events)-1] != "data: [DONE]" {
		t.Fatalf("last event %q", events[len(events)-1])
	}
	for _, e := range events[:len(events)-1] {
		var chunk struct {
			Object  string `json:"object"`
			Choices []struct {
				Delta struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := vibejson.Unmarshal([]byte(strings.TrimPrefix(e, "data: ")), &chunk); err != nil || chunk.Object != "chat.completion.chunk" {
			t.Fatalf("event %q: %v", e, err)
		}
		text += chunk.Choices[0].Delta.Content
		if f := chunk.Choices[0].FinishReason; f != nil {
			finish = *f
		}
	}
	if text != "You said: hello" || finish != "stop" || len(events) < 6 {
		t.Fatalf("%d events, text %q, finish %q", len(events), text, finish)
	}
	w = serve(s, http.MethodPost, "/v1/chat/completions", "application/json", []byte(`{"stream":true,"messages":[{"role":"user","content":"the time?"}],
		"tools":[{"type":"function","function":{"name":"now"}}]}`))
	if body := w.Body.String(); !strings.Contains(body, `"tool_calls":[{"index":0,"id":"call_`) || !strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Fatalf("streamed calls: %s", body)
	}
}

// The speech endpoint speaks the input as a WAV file, or streams PCM.
func TestSpeech(t *testing.T) {
	s := fakeServer(t)
	w := serve(s, http.MethodPost, "/v1/audio/speech", "application/json", []byte(`{"model":"judge","input":"Hello.","voice":"alloy","instructions":"Calmly."}`))
	// Tone: 10 ms a byte at 24 kHz, 240 samples, 16 bits each.
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "audio/wav" || w.Body.Len() != 44+6*240*2 || string(w.Body.Bytes()[:4]) != "RIFF" {
		t.Fatalf("wav: status %d, %s, %d bytes", w.Code, w.Header().Get("Content-Type"), w.Body.Len())
	}
	if rate := binary.LittleEndian.Uint32(w.Body.Bytes()[24:]); rate != 24000 {
		t.Fatalf("wav at %d Hz", rate)
	}
	w = serve(s, http.MethodPost, "/v1/audio/speech", "application/json", []byte(`{"input":"Hello.","response_format":"pcm"}`))
	if w.Code != http.StatusOK || w.Body.Len() != 6*240*2 {
		t.Fatalf("pcm: status %d, %d bytes", w.Code, w.Body.Len())
	}
	for _, bad := range []string{`{"input":""}`, `{"input":"x","response_format":"mp3"}`, `{"input":"x","language":"klingon"}`} {
		if w := serve(s, http.MethodPost, "/v1/audio/speech", "application/json", []byte(bad)); w.Code != http.StatusBadRequest {
			t.Fatalf("%s: status %d: %s", bad, w.Code, w.Body)
		}
	}
}
