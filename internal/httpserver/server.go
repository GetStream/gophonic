// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package httpserver serves an OpenAI-compatible transcription endpoint for
// any speech.Transcriber. Each worker lane owns bounded upload and inference
// scratch. Warm WAV requests reuse that storage; Go net/http still owns its
// transport allocations.
package httpserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"unsafe"

	"github.com/GetStream/gophonic/internal/audiofile"
	"github.com/GetStream/gophonic/internal/transcriptformat"
	"github.com/GetStream/gophonic/speech"
)

const MaxUploadBytes int64 = 25 << 20

// Server shares one immutable model across independent transcriber lanes.
// Close it only after the HTTP server has stopped accepting and draining requests.
type Server struct {
	mux        *http.ServeMux
	lanes      chan *worker
	admission  chan struct{}
	all        []*worker
	maxSeconds int
}

type worker struct {
	transcriber speech.Transcriber
	resampler   *speech.Resampler
	mono        []float32
	decoded     []float32
	upload      []byte
	response    []byte
	transcript  speech.Transcript
}

// NewServer constructs all inference lanes with newTranscriber before
// serving requests.
func NewServer(newTranscriber func() (speech.Transcriber, error), workers, maxSeconds int) (*Server, error) {
	if workers < 1 || workers > 64 {
		return nil, errors.New("workers must be between 1 and 64")
	}
	if maxSeconds < 1 || maxSeconds > 3600 {
		return nil, errors.New("max audio seconds must be between 1 and 3600")
	}
	s := &Server{mux: http.NewServeMux(), lanes: make(chan *worker, workers), admission: make(chan struct{}, workers*2), maxSeconds: maxSeconds}
	for range workers {
		transcriber, err := newTranscriber()
		if err != nil {
			s.Close()
			return nil, err
		}
		l := &worker{transcriber: transcriber, resampler: speech.NewResampler(), mono: make([]float32, maxSeconds*16000+1), upload: make([]byte, MaxUploadBytes+1), response: make([]byte, 0, 65536),
			transcript: speech.Transcript{Text: make([]byte, 0, 65536), Segments: make([]speech.Segment, 0, 512), Words: make([]speech.Word, 0, 4096)}}
		s.all = append(s.all, l)
		s.lanes <- l
	}
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	s.mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	s.mux.HandleFunc("POST /v1/audio/transcriptions", s.transcribe)
	return s, nil
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

// Close releases lane-owned workspaces after HTTP shutdown. It is not
// concurrent with ServeHTTP.
func (s *Server) Close() {
	if s == nil {
		return
	}
	for _, l := range s.all {
		l.resampler.Close()
		_ = l.transcriber.Close()
	}
	s.all = nil
}

func (s *Server) transcribe(w http.ResponseWriter, r *http.Request) {
	select {
	case s.admission <- struct{}{}:
		defer func() { <-s.admission }()
	default:
		writeError(w, http.StatusServiceUnavailable, "inference queue is full")
		return
	}
	var lane *worker
	select {
	case lane = <-s.lanes:
	case <-r.Context().Done():
		writeError(w, http.StatusRequestTimeout, "request canceled while waiting for inference")
		return
	}
	defer func() { s.lanes <- lane }()
	if r.ContentLength > MaxUploadBytes {
		writeError(w, http.StatusRequestEntityTooLarge, "upload exceeds 25 MiB")
		return
	}
	nbody, err := readBodyInto(r.Body, lane.upload)
	if err != nil {
		if errors.Is(err, errUploadTooLarge) {
			writeError(w, http.StatusRequestEntityTooLarge, "upload exceeds 25 MiB")
		} else {
			writeError(w, http.StatusBadRequest, "could not read upload")
		}
		return
	}
	submitted, err := parseUpload(r.Header.Get("Content-Type"), lane.upload[:nbody])
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if !bytes.HasSuffix(submitted.name, []byte(".wav")) && !bytes.HasSuffix(submitted.name, []byte(".WAV")) {
		writeError(w, http.StatusBadRequest, "HTTP uploads require WAV audio")
		return
	}
	pcm, rate, channels, err := audiofile.DecodeWAVInto(submitted.audio, lane.decoded, s.maxSeconds)
	if err == nil {
		lane.decoded = pcm[:0]
	}
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
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
	if count > len(lane.mono) {
		writeError(w, http.StatusBadRequest, "audio exceeds duration limit")
		return
	}
	n, err := lane.resampler.Resample16kInto(pcm, rate, channels, lane.mono[:count])
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	opts := speech.Options{Language: submitted.language, Segments: submitted.format >= 2, Words: submitted.words}
	if len(submitted.prompt) != 0 {
		// The prompt lives in the lane's upload buffer, which outlives the
		// call; viewing it as a string avoids a copy.
		opts.Context = unsafe.String(unsafe.SliceData(submitted.prompt), len(submitted.prompt))
	}
	t := &lane.transcript
	if err := lane.transcriber.Transcribe(r.Context(), lane.mono[:n], opts, t); err != nil {
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
		lane.response = transcriptformat.AppendSubtitles(lane.response[:0], t, submitted.format == 4)
		_, _ = w.Write(lane.response)
		return
	}
	w.Header()["Content-Type"] = jsonContentType
	if submitted.format == 2 {
		lane.response = transcriptformat.AppendVerboseJSON(lane.response[:0], t, submitted.words, n)
		_, _ = w.Write(lane.response)
		return
	}
	lane.response = append(lane.response[:0], '{', '"', 't', 'e', 'x', 't', '"', ':')
	lane.response = transcriptformat.AppendJSONString(lane.response, trimmed)
	lane.response = append(lane.response, '}', '\n')
	_, _ = w.Write(lane.response)
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

func writeError(w http.ResponseWriter, code int, message string) {
	w.Header()["Content-Type"] = jsonContentType
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}{Error: struct {
		Message string `json:"message"`
	}{Message: message}})
}
