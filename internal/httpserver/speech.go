// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package httpserver

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
	"net/http"
	"slices"
	"unsafe"

	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

// openAIVoices are the voices OpenAI clients name; they select the model's
// default voice.
var openAIVoices = []string{"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"}

// speechRequest is what gophonic serves of an OpenAI speech request.
type speechRequest struct {
	model, input, voice, instructions, format string
	language                                  speech.Language
}

func parseSpeech(body []byte, req *speechRequest) error {
	*req = speechRequest{format: "wav"}
	return vibejson.EachObject(body, func(key string, v vibejson.RawValue) error {
		switch key {
		case "model":
			req.model = text(v)
		case "input":
			req.input = text(v)
		case "voice":
			req.voice = text(v)
		case "instructions":
			req.instructions = text(v)
		case "response_format":
			req.format = text(v)
		case "language":
			l, ok := speech.ParseLanguage(text(v))
			if !ok {
				return errors.New("unknown language")
			}
			req.language = l
		}
		return nil
	})
}

// speak answers POST /v1/audio/speech as OpenAI's API does, with a
// speech.Synthesizer lane: the input is spoken in the voice named (an
// OpenAI voice picks the model's default), styled by the instructions. The
// response is a WAV file ("wav", the default), or 16-bit little-endian PCM
// at the voice's rate ("pcm"), which streams as it is decoded.
func (s *Server) speak(w http.ResponseWriter, r *http.Request) {
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
	var req speechRequest
	if err := parseSpeech(l.upload[:nbody], &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid speech request: "+err.Error())
		return
	}
	if req.input == "" {
		writeError(w, http.StatusBadRequest, "a speech request needs input")
		return
	}
	if req.format != "wav" && req.format != "pcm" {
		writeError(w, http.StatusBadRequest, "response_format "+req.format+" is not served: wav or pcm")
		return
	}
	lease, err := acquire[speech.Synthesizer](s, unsafe.Slice(unsafe.StringData(req.model), len(req.model)))
	if err != nil {
		writeLeaseError(w, err)
		return
	}
	defer lease.Release()
	voice := lease.Lane
	name := req.voice
	if slices.Contains(openAIVoices, name) && !slices.Contains(voice.Voices(), name) {
		name = ""
	}
	opts := speech.SpeakOptions{Voice: name, Language: req.language, Style: req.instructions}
	if err := voice.Begin(r.Context(), opts); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	voice.Write(unsafe.Slice(unsafe.StringData(req.input), len(req.input)))
	voice.End()
	rate := voice.SampleRate()
	limit := s.maxSeconds * rate // what one response may hold
	l.voice = slices.Grow(l.voice[:0], rate/10)[:rate/10]
	stream := req.format == "pcm"
	flusher, _ := w.(http.Flusher)
	if stream {
		w.Header()["Content-Type"] = pcmContentType
	}
	l.speech = l.speech[:0]
	if !stream {
		l.speech = append(l.speech, make([]byte, 44)...) // the header, once the length is known
	}
	samples := 0
	for {
		n, err := voice.Read(l.voice)
		if samples += n; samples > limit {
			if !stream {
				writeError(w, http.StatusBadRequest, "the speech is longer than the server's limit")
			}
			return
		}
		for _, v := range l.voice[:n] {
			l.speech = binary.LittleEndian.AppendUint16(l.speech, uint16(int16(math.Round(float64(max(-1, min(1, v))*32767)))))
		}
		if stream && n > 0 {
			if _, werr := w.Write(l.speech); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			l.speech = l.speech[:0]
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if !stream {
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}
	}
	if stream {
		return
	}
	putWAVHeader(l.speech, rate)
	w.Header()["Content-Type"] = wavContentType
	_, _ = w.Write(l.speech)
}

// putWAVHeader writes the 44-byte header of 16-bit mono PCM at rate over
// the start of wav, whose samples follow it.
func putWAVHeader(wav []byte, rate int) {
	data := len(wav) - 44
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], uint32(36+data))
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1) // PCM
	binary.LittleEndian.PutUint16(wav[22:], 1) // mono
	binary.LittleEndian.PutUint32(wav[24:], uint32(rate))
	binary.LittleEndian.PutUint32(wav[28:], uint32(2*rate))
	binary.LittleEndian.PutUint16(wav[32:], 2)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], uint32(data))
}

var (
	wavContentType = []string{"audio/wav"}
	pcmContentType = []string{"audio/pcm"}
)
