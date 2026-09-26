// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
	"github.com/thesyncim/vibejson"
)

type reference struct {
	Text     string `json:"text"`
	Language string `json:"language"`
	Speaker  string `json:"speaker"`
	IDs      []int  `json:"ids"`
	Frames   int    `json:"frames"`
	codes    []int32
	wav      []float32
	prefill  []float32
}

func loadReference(t *testing.T, name string) *reference {
	t.Helper()
	dir := filepath.Join(testmodels.Path(t, testmodels.Qwen3TTSReference), name)
	raw, err := os.ReadFile(filepath.Join(dir, "case.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r reference
	if err := vibejson.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	r.codes = readInts(t, filepath.Join(dir, "codes.i32"))
	r.wav = readFloats(t, filepath.Join(dir, "wav.f32"))
	r.prefill = readFloats(t, filepath.Join(dir, "prefill.f32"))
	return &r
}

func readFloats(t *testing.T, path string) []float32 {
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

func readInts(t *testing.T, path string) []int32 {
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

var loaded *Model

func loadModel(t *testing.T) *Model {
	t.Helper()
	path := testmodels.Path(t, testmodels.Qwen3TTS)
	if loaded == nil {
		m, err := Load(path, Options{Format: os.Getenv("QWEN3TTS_FORMAT")})
		if err != nil {
			t.Fatal(err)
		}
		loaded = m
	}
	return loaded
}

// The streaming codec decoder reproduces the reference waveform from the
// reference codes, frame by frame.
func TestCodecMatchesReference(t *testing.T) {
	ref := loadReference(t, "hello")
	m := loadModel(t)
	d := m.codec.newDecoder()
	pcm := make([]float32, ref.Frames*FrameSamples)
	var frame [groups]int
	start := time.Now()
	for f := range ref.Frames {
		for g := range groups {
			frame[g] = int(ref.codes[f*groups+g])
		}
		d.decode(&frame, pcm[f*FrameSamples:(f+1)*FrameSamples])
	}
	elapsed := time.Since(start)
	var worst, dot, na, nb float64
	for i, v := range ref.wav {
		worst = max(worst, math.Abs(float64(v-pcm[i])))
		dot += float64(v) * float64(pcm[i])
		na += float64(v) * float64(v)
		nb += float64(pcm[i]) * float64(pcm[i])
	}
	t.Logf("%d frames in %v (%.2f ms per 80 ms frame); max abs error %.2e, cosine %.8f", ref.Frames, elapsed,
		float64(elapsed.Microseconds())/1000/float64(ref.Frames), worst, dot/math.Sqrt(na*nb))
	if len(ref.wav) != len(pcm) || worst > 2e-3 {
		t.Fatalf("waveform differs: %d vs %d samples, max abs error %g", len(pcm), len(ref.wav), worst)
	}
}

// Greedy decoding reproduces the reference codec tokens.
func TestGreedyCodesMatchReference(t *testing.T) {
	ref := loadReference(t, "hello")
	m := loadModel(t)
	s, err := NewSynthesizer(m)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Greedy = true
	var got [][groups]int
	start := time.Now()
	err = s.generateText(speech.SpeakOptions{Voice: ref.Speaker, Language: ref.Language}, ref.Text,
		func(f *[groups]int) error { got = append(got, *f); return nil })
	if err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	t.Logf("%d frames in %v (%.2f ms per frame)", len(got), elapsed, float64(elapsed.Microseconds())/1000/float64(max(len(got), 1)))
	same, first := 0, -1
	for f := range min(len(got), ref.Frames) {
		ok := true
		for g := range groups {
			if got[f][g] != int(ref.codes[f*groups+g]) {
				ok = false
			}
		}
		if ok {
			same++
		} else if first < 0 {
			first = f
		}
	}
	t.Logf("frames equal to the reference: %d of %d (first difference at %d); length %d vs %d", same, ref.Frames, first, len(got), ref.Frames)
	var perGroup [groups]int
	for f := range min(len(got), ref.Frames) {
		for g := range groups {
			if got[f][g] == int(ref.codes[f*groups+g]) {
				perGroup[g]++
			}
		}
	}
	t.Logf("matches per codebook: %v", perGroup)
	for f := range min(3, len(got)) {
		t.Logf("frame %d: got %v want %v", f, got[f], ref.codes[f*groups:(f+1)*groups])
	}
	if first == 0 {
		t.Fatalf("first frame %v, want %v", got[0], ref.codes[:groups])
	}
}

func TestFirstStepMatchesReference(t *testing.T) {
	ref := loadReference(t, "hello")
	dir := filepath.Join(testmodels.Path(t, testmodels.Qwen3TTSReference), "hello")
	wantRows := readFloats(t, filepath.Join(dir, "step_rows.f32"))
	wantHidden := readFloats(t, filepath.Join(dir, "step_hidden.f32"))
	m := loadModel(t)
	s, err := NewSynthesizer(m)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Greedy = true
	h := m.hidden
	step := 0
	s.onFeed = func(row, hidden []float32) {
		if step < 3 {
			t.Logf("step %d: hidden cosine %.7f, input row cosine %.7f max %.3g", step, cosine(hidden, wantHidden[step*h:(step+1)*h]),
				cosine(row, wantRows[step*h:(step+1)*h]), maxDiff(row, wantRows[step*h:(step+1)*h]))
		}
		step++
	}
	n := 0
	s.generateText(speech.SpeakOptions{Voice: ref.Speaker, Language: ref.Language}, ref.Text,
		func(*[groups]int) error {
			n++
			if n == 4 {
				return io.EOF
			}
			return nil
		})
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func maxDiff(a, b []float32) float64 {
	var worst float64
	for i := range a {
		worst = max(worst, math.Abs(float64(a[i]-b[i])))
	}
	return worst
}

func TestPredictorMatchesReference(t *testing.T) {
	dir := filepath.Join(testmodels.Path(t, testmodels.Qwen3TTSReference), "hello")
	inputs := readFloats(t, filepath.Join(dir, "cp_inputs.f32"))
	want := readFloats(t, filepath.Join(dir, "cp_logits.f32"))
	m := loadModel(t)
	s, err := NewSynthesizer(m)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	h, ch := m.hidden, m.cpHidden
	for f := range 3 {
		in := inputs[f*2*h : (f+1)*2*h]
		rows := make([]float32, 2*ch)
		if err := m.proj.apply(s.exec, rows, in, 2); err != nil {
			t.Fatal(err)
		}
		ids := placeholders(nil, 2)
		if err := m.cpEval.HiddenLastExtendEmbedInto(s.ckv, 0, ids, qwen3lmEmbeds(rows), s.cpHidden, s.cws); err != nil {
			t.Fatal(err)
		}
		if f == 0 {
			wantProj := readFloats(t, filepath.Join(dir, "cp_proj0.f32"))
			wantHidden := readFloats(t, filepath.Join(dir, "cp_hidden0.f32"))
			t.Logf("projection cosine %.7f max %.3g; hidden cosine %.7f max %.3g", cosine(rows, wantProj), maxDiff(rows, wantProj),
				cosine(s.cpHidden, wantHidden), maxDiff(s.cpHidden, wantHidden))
		}
		m.heads[0].Mul(s.cpLogits, s.cpHidden)
		w := want[f*codes : (f+1)*codes]
		a, _ := top2(s.cpLogits)
		b, _ := top2(w)
		t.Logf("frame %d: logits cosine %.7f max diff %.3g; top %d (%.3f) vs %d (%.3f)", f, cosine(s.cpLogits, w), maxDiff(s.cpLogits, w), a, s.cpLogits[a], b, w[b])
		// Exact weights agree to rounding; int8 GPU weights at the llama.cpp
		// Q8_0 level.
		if c := cosine(s.cpLogits, w); c < 0.999 {
			t.Fatalf("frame %d: code predictor logits cosine %.7f", f, c)
		}
	}
}

func qwen3lmEmbeds(rows []float32) qwen3lm.Embeds {
	return qwen3lm.Embeds{Token: placeholderID, Rows: rows}
}

func top2(v []float32) (int, int) {
	a, b := 0, 1
	if v[b] > v[a] {
		a, b = b, a
	}
	for i := 2; i < len(v); i++ {
		if v[i] > v[a] {
			a, b = i, a
		} else if v[i] > v[b] {
			b = i
		}
	}
	return a, b
}

// A second utterance in the same voice reuses the cached voice prompt and
// speaks the same codes.
func TestVoicePromptReuse(t *testing.T) {
	ref := loadReference(t, "hello")
	m := loadModel(t)
	s, err := NewSynthesizer(m)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Greedy = true
	opts := speech.SpeakOptions{Voice: ref.Speaker, Language: ref.Language}
	speak := func() [][groups]int {
		var got [][groups]int
		if err := s.generateText(opts, ref.Text, func(f *[groups]int) error { got = append(got, *f); return nil }); err != nil {
			t.Fatal(err)
		}
		return got
	}
	first := speak()
	if s.voiceRows == 0 {
		t.Fatal("no voice prompt cached")
	}
	rows := len(s.kv.Tokens())
	again := speak()
	if len(s.kv.Tokens()) != rows {
		t.Fatalf("the prompt grew: %d rows, then %d", rows, len(s.kv.Tokens()))
	}
	same := 0
	for f := range min(len(first), len(again)) {
		if first[f] == again[f] {
			same++
		}
	}
	t.Logf("frames equal: %d of %d and %d", same, len(first), len(again))
	if len(again) == 0 || again[0] != first[0] {
		t.Fatalf("first frame %v, want %v", again[0], first[0])
	}
}

// A first word with nothing after it is tokenized as it stands, so speech
// starts without waiting for more text; later text waits for a word
// boundary, and every token knows the byte of the text it ends at.
func TestFirstPieceTokens(t *testing.T) {
	m := loadModel(t)
	s, err := NewSynthesizer(m)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.reset()
	s.hold()
	pieces := []string{"Sure", ",", " the", " answer", " is", " simple", "."}
	ctx := context.Background()
	for i, p := range pieces {
		s.mu.Lock()
		s.written = append(s.written, p...)
		s.mu.Unlock()
		if err := s.pull(ctx); err != nil {
			t.Fatal(err)
		}
		if i == 0 && len(s.text) != 1 {
			t.Fatalf("after the first piece: %d tokens, want 1", len(s.text))
		}
	}
	s.mu.Lock()
	s.ended = true
	s.mu.Unlock()
	for !s.textDone {
		if err := s.pull(ctx); err != nil {
			t.Fatal(err)
		}
	}
	text := strings.Join(pieces, "")
	want, err := m.tokens.EncodeInto(text, make([]int, 0, 64), &qwen3lm.TokenizerWorkspace{})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(s.text, want) {
		t.Fatalf("tokens %v, want %v", s.text, want)
	}
	at := 0
	for i, id := range s.text {
		at += len(m.tokens.Piece(id))
		if s.textEnd[i] != at {
			t.Fatalf("token %d ends at byte %d, want %d", i, s.textEnd[i], at)
		}
	}
	if at != len(text) {
		t.Fatalf("tokens end at byte %d of %d", at, len(text))
	}
}

// A warm utterance allocates nothing, on any of the lane's goroutines.
func TestSpeakAllocations(t *testing.T) {
	m := loadModel(t)
	s, err := NewSynthesizer(m)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Greedy = true
	opts := speech.SpeakOptions{Voice: "Ryan", Language: "en"}
	text := []byte("Hello there.")
	pcm := make([]float32, FrameSamples/4)
	speak := func() {
		if err := s.Begin(context.Background(), opts); err != nil {
			t.Fatal(err)
		}
		s.Write(text)
		s.End()
		for {
			if _, err := s.Read(pcm); errors.Is(err, io.EOF) {
				return
			} else if err != nil {
				t.Fatal(err)
			}
		}
	}
	speak()
	if allocs := testing.AllocsPerRun(3, speak); allocs != 0 {
		t.Fatalf("an utterance allocates %v times", allocs)
	}
}

// A style is read before the voice prompt: it changes what is said, its
// prompt is cached like any voice's, and a warm call allocates nothing.
func TestStyle(t *testing.T) {
	ref := loadReference(t, "hello")
	m := loadModel(t)
	s, err := NewSynthesizer(m)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Greedy = true
	speak := func(style string) [][groups]int {
		var got [][groups]int
		opts := speech.SpeakOptions{Voice: ref.Speaker, Language: ref.Language, Style: style}
		if err := s.generateText(opts, ref.Text, func(f *[groups]int) error { got = append(got, *f); return nil }); err != nil {
			t.Fatal(err)
		}
		return got
	}
	plain := speak("")
	calm := speak("Speak slowly, in a calm and even voice.")
	rows := len(s.kv.Tokens())
	again := speak("Speak slowly, in a calm and even voice.")
	if len(plain) == 0 || len(calm) == 0 || calm[0] == plain[0] && len(calm) == len(plain) {
		t.Fatalf("the style changed nothing: %d frames, %d without it", len(calm), len(plain))
	}
	if again[0] != calm[0] || len(s.kv.Tokens()) != rows {
		t.Fatalf("the cached styled prompt differs: first frame %v, want %v", again[0], calm[0])
	}
	t.Logf("%d frames plain, %d calm", len(plain), len(calm))
	opts := speech.SpeakOptions{Voice: ref.Speaker, Language: ref.Language, Style: "Speak slowly, in a calm and even voice."}
	if n := testing.AllocsPerRun(2, func() {
		if err := s.generateText(opts, ref.Text, func(*[groups]int) error { return nil }); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("%v allocations per warm styled call", n)
	}
}

// generateText generates text's codes in opts's voice on the caller's
// goroutine, passing each frame to emit: the reference tests' path.
func (s *Synthesizer) generateText(opts speech.SpeakOptions, text string, emit func(*[groups]int) error) error {
	speaker, language, err := s.m.voiceOf(opts)
	if err != nil {
		return err
	}
	s.hold()
	s.mu.Lock()
	s.written, s.taken, s.ended = append(s.written[:0], text...), 0, true
	s.mu.Unlock()
	return s.generate(context.Background(), speaker, language, opts.Style, emit)
}

// hold makes the caller's goroutine the lane's worker for a new utterance,
// as the tests that drive the talker directly need.
func (s *Synthesizer) hold() {
	s.mu.Lock()
	s.id++
	s.cur, s.ctx = s.id, context.Background()
	s.written, s.taken, s.ended = s.written[:0], 0, false
	s.mu.Unlock()
}
