// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package httpserver

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GetStream/gophonic/whisper"
)

func multipartRequest(t *testing.T, filename string, audio []byte, fields map[string]string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for name, value := range fields {
		if err := writer.WriteField(name, value); err != nil {
			t.Fatal(err)
		}
	}
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(audio); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	return request
}

func parseTestUpload(t *testing.T, r *http.Request) (upload, error) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}
	return parseUpload(r.Header.Get("Content-Type"), body)
}

func TestReadUploadValidatesAPIFields(t *testing.T) {
	request := multipartRequest(t, "sample.wav", []byte("wav"), map[string]string{
		"model": "gophonic-whisper", "response_format": "text", "language": "en",
	})
	got, err := parseTestUpload(t, request)
	if err != nil || string(got.name) != "sample.wav" || string(got.audio) != "wav" || !got.text {
		t.Fatalf("upload: %+v err=%v", got, err)
	}
	limited := multipartRequest(t, "sample.wav", []byte("wav"), nil)
	small := make([]byte, 16)
	if _, err := readBodyInto(limited.Body, small); !errors.Is(err, errUploadTooLarge) {
		t.Fatalf("oversized upload error = %v", err)
	}
	for _, fields := range []map[string]string{
		{"model": "unsupported"}, {"language": "fr"},
		{"response_format": "verbose_json"}, {"prompt": "not implemented"},
	} {
		request := multipartRequest(t, "sample.wav", []byte("wav"), fields)
		if _, err := parseTestUpload(t, request); err == nil {
			t.Fatalf("unsupported fields accepted: %v", fields)
		}
	}
}

func TestWhisperServerOfficialJFK(t *testing.T) {
	modelPath := os.Getenv("GOPHONIC_WHISPER_MODEL")
	if modelPath == "" {
		t.Skip("set GOPHONIC_WHISPER_MODEL to the official converted tiny.en bundle")
	}
	model, err := whisper.Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewWhisperServer(model, 1, 120)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()

	pcm, err := os.ReadFile(filepath.Join("..", "..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	var wav bytes.Buffer
	wav.WriteString("RIFF")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(36+len(pcm)))
	wav.WriteString("WAVEfmt ")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(3))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(1))
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16000))
	_ = binary.Write(&wav, binary.LittleEndian, uint32(64000))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(4))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(32))
	wav.WriteString("data")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(len(pcm)))
	_, _ = wav.Write(pcm)

	for _, tc := range []struct{ format, wantType string }{{"json", "application/json"}, {"text", "text/plain"}} {
		request := multipartRequest(t, "jfk.wav", wav.Bytes(), map[string]string{"model": "gophonic-whisper", "response_format": tc.format})
		response := httptest.NewRecorder()
		server.ServeHTTP(response, request)
		if response.Code != http.StatusOK || !strings.HasPrefix(response.Header().Get("Content-Type"), tc.wantType) {
			t.Fatalf("%s response: status=%d content-type=%q body=%q", tc.format, response.Code, response.Header().Get("Content-Type"), response.Body.String())
		}
		var transcript string
		if tc.format == "json" {
			var result struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			transcript = result.Text
		} else {
			transcript = response.Body.String()
		}
		if transcript != "And so my fellow Americans ask not what your country can do for you ask what you can do for your country." {
			t.Fatalf("unexpected transcript %q", transcript)
		}
	}
	bad := httptest.NewRecorder()
	server.ServeHTTP(bad, multipartRequest(t, "invalid.wav", []byte("not WAV"), nil))
	if bad.Code != http.StatusBadRequest {
		t.Fatalf("invalid WAV status=%d body=%q", bad.Code, bad.Body.String())
	}
	for _, route := range []string{"/healthz", "/readyz"} {
		response := httptest.NewRecorder()
		server.ServeHTTP(response, httptest.NewRequest(http.MethodGet, route, nil))
		if response.Code != http.StatusOK {
			t.Fatalf("%s status=%d", route, response.Code)
		}
	}
}

type discardResponse struct {
	header http.Header
	status int
	bytes  int
}

func (w *discardResponse) Header() http.Header         { return w.header }
func (w *discardResponse) WriteHeader(code int)        { w.status = code }
func (w *discardResponse) Write(p []byte) (int, error) { w.bytes += len(p); return len(p), nil }

func TestMultipartParserZeroAlloc(t *testing.T) {
	request := multipartRequest(t, "sample.wav", []byte("wav"), map[string]string{"model": "gophonic-whisper"})
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatal(err)
	}
	contentType := request.Header.Get("Content-Type")
	allocs := testing.AllocsPerRun(100, func() {
		got, err := parseUpload(contentType, body)
		if err != nil || len(got.audio) != 3 {
			panic("parse failed")
		}
	})
	if allocs != 0 {
		t.Fatalf("multipart parser: %g allocs/op", allocs)
	}
}

func TestWarmedWAVHandlerAllocations(t *testing.T) {
	modelPath := os.Getenv("GOPHONIC_WHISPER_MODEL")
	if modelPath == "" {
		t.Skip("set GOPHONIC_WHISPER_MODEL to measure handler allocations")
	}
	model, err := whisper.Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewWhisperServer(model, 1, 5)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	pcm, err := os.ReadFile(filepath.Join("..", "..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	pcm = pcm[:min(len(pcm), 16000*4)]
	wav := make([]byte, 44+len(pcm))
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
	binary.LittleEndian.PutUint32(wav[40:], uint32(len(pcm)))
	copy(wav[44:], pcm)
	request := multipartRequest(t, "jfk.wav", wav, nil)
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatal(err)
	}
	reader := bytes.NewReader(body)
	request.Body = io.NopCloser(reader)
	response := &discardResponse{header: make(http.Header)}
	call := func() {
		reader.Reset(body)
		response.status, response.bytes = 0, 0
		server.transcribe(response, request)
		if response.status != 0 || response.bytes == 0 {
			panic("transcription failed")
		}
	}
	call()
	allocs := testing.AllocsPerRun(3, call)
	t.Logf("warmed WAV handler allocations: %g allocs/op", allocs)
	if allocs != 0 {
		t.Fatalf("warmed WAV handler must allocate zero heap objects, got %g", allocs)
	}
}
