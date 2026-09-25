// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package httpserver serves gophonic models over HTTP. Transcription and
// the model list follow the OpenAI API, so OpenAI clients work unchanged;
// audio and text classification, which OpenAI does not offer, follow its
// conventions. Models load on first use from a gophonic.Pool, which closes
// them when idle. Each request slot owns bounded upload and inference
// scratch, and warm WAV requests reuse it; Go net/http still owns its
// transport allocations.
package httpserver

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"unsafe"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/internal/audiofile"
	"github.com/GetStream/gophonic/internal/transcriptformat"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

const MaxUploadBytes int64 = 25 << 20

// Model is a model the server serves.
type Model struct {
	// Name is what requests call it: their "model" field.
	Name string
	// Path is what the pool opens.
	Path string
	// Format is how gophonic opens Path, as gophonic.Detect reports.
	Format gophonic.Format
}

// Config describes a Server.
type Config struct {
	// Pool opens the models and closes them when idle.
	Pool *gophonic.Pool
	// Models are the models served. A request that names no model, or an
	// OpenAI model such as "whisper-1", gets the first that can serve it.
	Models []Model
	// Workers bounds the requests running at once; as many more may wait.
	Workers int
	// MaxSeconds bounds the audio of one request.
	MaxSeconds int
}

// Server serves a set of models. Close it only after the HTTP server has
// stopped accepting and draining requests.
type Server struct {
	mux        *http.ServeMux
	pool       *gophonic.Pool
	models     []Model
	byName     map[string]int
	slots      chan *slot
	admission  chan struct{}
	all        []*slot
	maxSeconds int
	questions  questions

	mu       sync.Mutex
	defaults map[reflect.Type]int // the model serving each lane type by default
}

// slot is one request's scratch.
type slot struct {
	resampler  *speech.Resampler
	mono       []float32
	decoded    []float32
	upload     []byte
	response   []byte
	transcript speech.Transcript
	probs      []float32
	text       textRequest
	texts      textsRequest
}

// NewServer returns a server of cfg's models. It opens no model.
func NewServer(cfg Config) (*Server, error) {
	if cfg.Workers < 1 || cfg.Workers > 64 {
		return nil, errors.New("workers must be between 1 and 64")
	}
	if cfg.MaxSeconds < 1 || cfg.MaxSeconds > 3600 {
		return nil, errors.New("max audio seconds must be between 1 and 3600")
	}
	if cfg.Pool == nil || len(cfg.Models) == 0 {
		return nil, errors.New("a server needs a pool and at least one model")
	}
	s := &Server{mux: http.NewServeMux(), pool: cfg.Pool, models: cfg.Models, byName: map[string]int{},
		slots: make(chan *slot, cfg.Workers), admission: make(chan struct{}, cfg.Workers*2),
		maxSeconds: cfg.MaxSeconds, defaults: map[reflect.Type]int{}}
	for i, m := range cfg.Models {
		if _, dup := s.byName[m.Name]; dup {
			return nil, fmt.Errorf("two models are named %q", m.Name)
		}
		s.byName[m.Name] = i
	}
	for range cfg.Workers {
		l := &slot{resampler: speech.NewResampler(), mono: make([]float32, cfg.MaxSeconds*16000+1), upload: make([]byte, MaxUploadBytes+1), response: make([]byte, 0, 65536),
			transcript: speech.Transcript{Text: make([]byte, 0, 65536), Segments: make([]speech.Segment, 0, 512), Words: make([]speech.Word, 0, 4096)}}
		s.all = append(s.all, l)
		s.slots <- l
	}
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	s.mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	s.mux.HandleFunc("GET /v1/models", s.listModels)
	s.mux.HandleFunc("POST /v1/audio/transcriptions", s.transcribe)
	s.mux.HandleFunc("POST /v1/audio/classifications", s.classifyAudio)
	s.mux.HandleFunc("POST /v1/classifications", s.classifyText)
	return s, nil
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

// Close releases the request scratch and prepared classifiers after HTTP
// shutdown; the pool, which the caller owns, closes the models. It is not
// concurrent with ServeHTTP.
func (s *Server) Close() {
	if s == nil {
		return
	}
	for _, l := range s.all {
		l.resampler.Close()
	}
	s.all = nil
	s.questions.close()
}

// openAIModels are names OpenAI clients send; they select the default model.
var openAIModels = []string{"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe-diarize"}

func isOpenAIModel(name []byte) bool {
	for _, m := range openAIModels {
		if string(name) == m {
			return true
		}
	}
	return false
}

// errNoModel reports a request for a model the server does not serve.
type errNoModel struct{ name string }

func (e errNoModel) Error() string { return "model " + strconv.Quote(e.name) + " is not served here" }

// acquire leases a lane of type T from the model a request names, or from
// the default model for T when it names none or an OpenAI model.
func acquire[T any](s *Server, name []byte) (gophonic.Lease[T], error) {
	if i, ok := s.byName[string(name)]; ok {
		return gophonic.Acquire[T](s.pool, s.models[i].Path)
	}
	if len(name) != 0 && !isOpenAIModel(name) {
		return gophonic.Lease[T]{}, errNoModel{string(name)}
	}
	typ := reflect.TypeFor[T]()
	s.mu.Lock()
	i, ok := s.defaults[typ]
	s.mu.Unlock()
	if ok {
		return gophonic.Acquire[T](s.pool, s.models[i].Path)
	}
	// Models whose format declares T come first; those that declare
	// nothing are opened to find out.
	var first error
	for pass := range 2 {
		for i, m := range s.models {
			declared := m.Format.Provides != nil
			if pass == 0 && !slices.Contains(m.Format.Provides, typ) || pass == 1 && declared {
				continue
			}
			lease, err := gophonic.Acquire[T](s.pool, m.Path)
			if errors.Is(err, speech.ErrUnsupported) {
				continue
			}
			if err != nil {
				first = cmp(first, err)
				continue
			}
			s.mu.Lock()
			s.defaults[typ] = i
			s.mu.Unlock()
			return lease, nil
		}
	}
	if first != nil {
		return gophonic.Lease[T]{}, first
	}
	return gophonic.Lease[T]{}, fmt.Errorf("no model here provides %v: %w", typ, speech.ErrUnsupported)
}

func cmp(a, b error) error {
	if a != nil {
		return a
	}
	return b
}

// writeLeaseError reports why a lane could not be leased.
func writeLeaseError(w http.ResponseWriter, err error) {
	var missing errNoModel
	switch {
	case errors.As(err, &missing):
		writeError(w, http.StatusNotFound, err.Error())
	case errors.Is(err, speech.ErrUnsupported):
		writeError(w, http.StatusBadRequest, err.Error())
	default:
		writeError(w, http.StatusInternalServerError, "could not load the model: "+err.Error())
	}
}

// admit takes a request slot, or reports why none is available.
func (s *Server) admit(w http.ResponseWriter, r *http.Request) (*slot, bool) {
	select {
	case s.admission <- struct{}{}:
	default:
		writeError(w, http.StatusServiceUnavailable, "inference queue is full")
		return nil, false
	}
	select {
	case l := <-s.slots:
		return l, true
	case <-r.Context().Done():
		<-s.admission
		writeError(w, http.StatusRequestTimeout, "request canceled while waiting for inference")
		return nil, false
	}
}

func (s *Server) done(l *slot) {
	s.slots <- l
	<-s.admission
}

// readUpload reads and parses a multipart upload with a WAV file.
func (s *Server) readUpload(w http.ResponseWriter, r *http.Request, l *slot) (upload, bool) {
	if r.ContentLength > MaxUploadBytes {
		writeError(w, http.StatusRequestEntityTooLarge, "upload exceeds 25 MiB")
		return upload{}, false
	}
	nbody, err := readBodyInto(r.Body, l.upload)
	if err != nil {
		if errors.Is(err, errUploadTooLarge) {
			writeError(w, http.StatusRequestEntityTooLarge, "upload exceeds 25 MiB")
		} else {
			writeError(w, http.StatusBadRequest, "could not read upload")
		}
		return upload{}, false
	}
	submitted, err := parseUpload(r.Header.Get("Content-Type"), l.upload[:nbody])
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return upload{}, false
	}
	if !bytes.HasSuffix(submitted.name, []byte(".wav")) && !bytes.HasSuffix(submitted.name, []byte(".WAV")) {
		writeError(w, http.StatusBadRequest, "HTTP uploads require WAV audio")
		return upload{}, false
	}
	return submitted, true
}

// decodeWAV decodes an upload's audio into the slot.
func (s *Server) decodeWAV(w http.ResponseWriter, l *slot, audio []byte) (pcm []float32, rate, channels int, ok bool) {
	pcm, rate, channels, err := audiofile.DecodeWAVInto(audio, l.decoded, s.maxSeconds)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return nil, 0, 0, false
	}
	l.decoded = pcm[:0]
	return pcm, rate, channels, true
}

func (s *Server) transcribe(w http.ResponseWriter, r *http.Request) {
	l, ok := s.admit(w, r)
	if !ok {
		return
	}
	defer s.done(l)
	submitted, ok := s.readUpload(w, r, l)
	if !ok {
		return
	}
	pcm, rate, channels, ok := s.decodeWAV(w, l, submitted.audio)
	if !ok {
		return
	}
	if r.Context().Err() != nil {
		writeError(w, http.StatusRequestTimeout, "request canceled before inference")
		return
	}
	count, err := speech.Samples16k(len(pcm), rate, channels)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if count > len(l.mono) {
		writeError(w, http.StatusBadRequest, "audio exceeds duration limit")
		return
	}
	n, err := l.resampler.Resample16kInto(pcm, rate, channels, l.mono[:count])
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	lease, err := acquire[speech.Transcriber](s, submitted.model)
	if err != nil {
		writeLeaseError(w, err)
		return
	}
	defer lease.Release()
	opts := speech.Options{Language: submitted.language, Segments: submitted.format >= 2, Words: submitted.words}
	if len(submitted.prompt) != 0 {
		// The prompt lives in the slot's upload buffer, which outlives the
		// call; viewing it as a string avoids a copy.
		opts.Context = unsafe.String(unsafe.SliceData(submitted.prompt), len(submitted.prompt))
	}
	t := &l.transcript
	if err := lease.Lane.Transcribe(r.Context(), l.mono[:n], opts, t); err != nil {
		if errors.Is(err, speech.ErrUnsupported) {
			writeError(w, http.StatusBadRequest, err.Error())
		} else {
			writeError(w, http.StatusInternalServerError, "transcription failed")
		}
		return
	}
	trimmed := bytes.TrimSpace(t.Text)
	if submitted.format == 1 {
		w.Header()["Content-Type"] = textContentType
		_, _ = w.Write(trimmed)
		return
	}
	if submitted.format == 3 || submitted.format == 4 {
		if submitted.format == 3 {
			w.Header()["Content-Type"] = srtContentType
		} else {
			w.Header()["Content-Type"] = vttContentType
		}
		l.response = transcriptformat.AppendSubtitles(l.response[:0], t, submitted.format == 4)
		_, _ = w.Write(l.response)
		return
	}
	w.Header()["Content-Type"] = jsonContentType
	if submitted.format == 2 {
		l.response = transcriptformat.AppendVerboseJSON(l.response[:0], t, submitted.words, n)
		_, _ = w.Write(l.response)
		return
	}
	l.response = append(l.response[:0], `{"text":`...)
	l.response = transcriptformat.AppendJSONString(l.response, trimmed)
	l.response = append(l.response, '}', '\n')
	_, _ = w.Write(l.response)
}

// classifyAudio answers POST /v1/audio/classifications: a multipart WAV
// upload, as for transcription, scored by an audio classifier such as a
// turn detector. The reply lists each label's probability:
//
//	{"model":"smart-turn-v3.2","classes":[{"label":"incomplete","probability":0.03},...]}
func (s *Server) classifyAudio(w http.ResponseWriter, r *http.Request) {
	l, ok := s.admit(w, r)
	if !ok {
		return
	}
	defer s.done(l)
	submitted, ok := s.readUpload(w, r, l)
	if !ok {
		return
	}
	pcm, rate, channels, ok := s.decodeWAV(w, l, submitted.audio)
	if !ok {
		return
	}
	lease, err := acquire[speech.AudioClassifier](s, submitted.model)
	if err != nil {
		writeLeaseError(w, err)
		return
	}
	defer lease.Release()
	labels := lease.Lane.Labels()
	l.probs = slices.Grow(l.probs[:0], len(labels))[:len(labels)]
	if err := lease.Lane.ClassifyInto(pcm, rate, channels, l.probs); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	l.response = append(l.response[:0], `{"model":`...)
	l.response = transcriptformat.AppendJSONString(l.response, s.modelName(lease.Path()))
	l.response = append(l.response, `,"classes":`...)
	l.response = appendClasses(l.response, labels, l.probs)
	l.response = append(l.response, '}', '\n')
	w.Header()["Content-Type"] = jsonContentType
	_, _ = w.Write(l.response)
}

var (
	textDecoder, _  = vibejson.CompileDecoder[textRequest](vibejson.DecoderOptions{ZeroCopy: true, Replace: true})
	textsDecoder, _ = vibejson.CompileDecoder[textsRequest](vibejson.DecoderOptions{ZeroCopy: true, Replace: true})
)

// jsonGetRaw returns the raw JSON value at pointer in body.
func jsonGetRaw(body []byte, pointer string) ([]byte, bool, error) {
	raw, ok, err := vibejson.GetRaw(body, pointer)
	return bytes.TrimSpace(raw.Src), ok, err
}

// textRequest is the body of POST /v1/classifications with one input;
// textsRequest has several.
type textRequest struct {
	Model    string   `json:"model"`
	Input    string   `json:"input"`
	Question string   `json:"question"`
	Labels   []string `json:"labels"`
}

type textsRequest struct {
	Model    string   `json:"model"`
	Input    []string `json:"input"`
	Question string   `json:"question"`
	Labels   []string `json:"labels"`
}

// classifyText answers POST /v1/classifications. The JSON body names the
// input text, or a list of texts, and either a question with the labels
// that answer it, for a zero-shot model (a language model), or neither,
// for a text classifier with labels of its own:
//
//	{"model":"Qwen3-1.7B","input":"I love it","question":"What is the sentiment?","labels":["positive","negative"]}
//
// The reply has one result per input:
//
//	{"model":"Qwen3-1.7B","results":[{"classes":[{"label":"positive","probability":0.98},...]}]}
//
// A zero-shot model evaluates a question once and keeps it prepared for
// the requests that repeat it.
func (s *Server) classifyText(w http.ResponseWriter, r *http.Request) {
	l, ok := s.admit(w, r)
	if !ok {
		return
	}
	defer s.done(l)
	nbody, err := readBodyInto(r.Body, l.upload[:1<<20])
	if err != nil {
		writeError(w, http.StatusBadRequest, "could not read a JSON body of at most 1 MiB")
		return
	}
	body := l.upload[:nbody]
	raw, found, err := jsonGetRaw(body, "/input")
	if err != nil || !found {
		writeError(w, http.StatusBadRequest, "the body must be a JSON object with an input")
		return
	}
	req := &l.texts
	if bytes.HasPrefix(raw, []byte("[")) {
		err = textsDecoder.Decode(body, req)
	} else if err = textDecoder.Decode(body, &l.text); err == nil {
		req.Model, req.Question, req.Labels = l.text.Model, l.text.Question, l.text.Labels
		req.Input = append(req.Input[:0], l.text.Input)
	}
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid classification request: "+err.Error())
		return
	}
	var (
		classifier speech.TextClassifier
		path       string
	)
	if req.Question == "" && len(req.Labels) == 0 {
		lease, err := acquire[speech.TextClassifier](s, unsafe.Slice(unsafe.StringData(req.Model), len(req.Model)))
		if err != nil {
			writeLeaseError(w, err)
			return
		}
		defer lease.Release()
		classifier, path = lease.Lane, lease.Path()
	} else {
		if req.Question == "" || len(req.Labels) < 2 {
			writeError(w, http.StatusBadRequest, "a zero-shot classification needs a question and at least two labels")
			return
		}
		lease, err := acquire[speech.ZeroShot](s, unsafe.Slice(unsafe.StringData(req.Model), len(req.Model)))
		if err != nil {
			writeLeaseError(w, err)
			return
		}
		defer lease.Release()
		path = lease.Path()
		q, err := s.questions.get(lease.Model(), path, lease.Lane, req.Question, req.Labels)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		defer q.mu.Unlock()
		classifier = q.c
	}
	labels := classifier.Labels()
	l.probs = slices.Grow(l.probs[:0], len(labels))[:len(labels)]
	l.response = append(l.response[:0], `{"model":`...)
	l.response = transcriptformat.AppendJSONString(l.response, s.modelName(path))
	l.response = append(l.response, `,"results":[`...)
	for i, text := range req.Input {
		if err := classifier.ClassifyInto(r.Context(), text, l.probs); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if i > 0 {
			l.response = append(l.response, ',')
		}
		l.response = append(l.response, `{"classes":`...)
		l.response = appendClasses(l.response, labels, l.probs)
		l.response = append(l.response, '}')
	}
	l.response = append(l.response, ']', '}', '\n')
	w.Header()["Content-Type"] = jsonContentType
	_, _ = w.Write(l.response)
}

// modelName returns the served name of the model at path.
func (s *Server) modelName(path string) []byte {
	for i := range s.models {
		if s.models[i].Path == path {
			return unsafe.Slice(unsafe.StringData(s.models[i].Name), len(s.models[i].Name))
		}
	}
	return nil
}

// appendClasses appends a JSON list of labels with their probabilities.
func appendClasses(dst []byte, labels []string, probs []float32) []byte {
	dst = append(dst, '[')
	for i, label := range labels {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = append(dst, `{"label":`...)
		dst = transcriptformat.AppendJSONString(dst, unsafe.Slice(unsafe.StringData(label), len(label)))
		dst = append(dst, `,"probability":`...)
		dst = strconv.AppendFloat(dst, float64(probs[i]), 'g', 6, 32)
		dst = append(dst, '}')
	}
	return append(dst, ']')
}

// modelList is the body of GET /v1/models: OpenAI's, plus each model's
// format, the lane types it provides when known, and whether it is loaded.
type modelList struct {
	Object string       `json:"object"`
	Data   []modelEntry `json:"data"`
}

type modelEntry struct {
	ID       string   `json:"id"`
	Object   string   `json:"object"`
	Created  int64    `json:"created"`
	OwnedBy  string   `json:"owned_by"`
	Format   string   `json:"format"`
	Provides []string `json:"provides,omitempty"`
	Loaded   bool     `json:"loaded"`
}

type errorBody struct {
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

var (
	modelListEncoder, _ = vibejson.CompileEncoder[modelList](vibejson.EncoderOptions{})
	errorEncoder, _     = vibejson.CompileEncoder[errorBody](vibejson.EncoderOptions{})
)

func (s *Server) listModels(w http.ResponseWriter, _ *http.Request) {
	list := modelList{Object: "list", Data: []modelEntry{}}
	loaded := s.pool.Loaded()
	for _, m := range s.models {
		e := modelEntry{ID: m.Name, Object: "model", OwnedBy: "gophonic", Format: m.Format.Name, Loaded: slices.Contains(loaded, m.Path)}
		if info, err := os.Stat(m.Path); err == nil {
			e.Created = info.ModTime().Unix()
		}
		for _, t := range m.Format.Provides {
			e.Provides = append(e.Provides, t.String())
		}
		list.Data = append(list.Data, e)
	}
	body, err := modelListEncoder.AppendJSON(nil, &list)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.Header()["Content-Type"] = jsonContentType
	_, _ = w.Write(append(body, '\n'))
}

var jsonContentType = []string{"application/json"}
var textContentType = []string{"text/plain; charset=utf-8"}
var srtContentType = []string{"application/x-subrip; charset=utf-8"}
var vttContentType = []string{"text/vtt; charset=utf-8"}

var errUploadTooLarge = errors.New("upload exceeds 25 MiB")

func readBodyInto(body io.Reader, dst []byte) (int, error) {
	n := 0
	for n < len(dst) {
		count, err := body.Read(dst[n:])
		n += count
		if err == io.EOF {
			return n, nil
		}
		if err != nil {
			return 0, err
		}
		if count == 0 {
			return 0, io.ErrNoProgress
		}
	}
	return 0, errUploadTooLarge
}

// writeError replies with an OpenAI error body.
func writeError(w http.ResponseWriter, code int, message string) {
	var e errorBody
	e.Error.Message = message
	body, _ := errorEncoder.AppendJSON(make([]byte, 0, 64+len(message)), &e)
	w.Header()["Content-Type"] = jsonContentType
	w.WriteHeader(code)
	_, _ = w.Write(append(body, '\n'))
}
