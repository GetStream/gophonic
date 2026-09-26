// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package httpserver

import (
	"bytes"
	"context"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/speech"
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
