// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3tts

import (
	"cmp"
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode"

	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/GetStream/gophonic/speech"
	"github.com/GetStream/gophonic/speech/speechtest"
	"github.com/GetStream/gophonic/whisper"
)

// A lane keeps the speech.Synthesizer contract.
func TestSynthesizerContract(t *testing.T) {
	s, err := NewSynthesizer(loadModel(t), LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	speechtest.TestSynthesizer(t, s, speech.SpeakOptions{Voice: "ryan", Language: speech.English},
		"Sure, the answer is simple: it rains because warm air cools as it rises.")
}

// Voiced follows the voice: as the audio is read, it reaches the word
// being spoken, by Whisper's timing of the words heard, within half a word
// on average and two at worst.
func TestVoicedFollowsWords(t *testing.T) {
	m := loadModel(t)
	if !m.aligned {
		t.Skip("no alignment head for this checkpoint")
	}
	ears := whisperEars(t)
	mean, worst := voicedDistance(t, m, ears, voicedSentences)
	t.Logf("Voiced is %.2f words from the word being spoken on average, %v at worst", mean, worst)
	if mean > 0.5 || worst > 2 {
		t.Fatalf("Voiced is %.2f words from the word being spoken on average, %v at worst", mean, worst)
	}
}

// TestRankAlignmentHeads finds a checkpoint's alignment head: ALIGN_RANK
// names a Qwen3-TTS checkpoint in the models directory, every talker head
// is ranked by how far Voiced, read from it, is from the word Whisper
// hears being spoken in one sentence, and the ten closest again in all of
// them.
func TestRankAlignmentHeads(t *testing.T) {
	name := os.Getenv("ALIGN_RANK")
	if name == "" {
		t.Skip("ALIGN_RANK names the checkpoint whose heads to rank")
	}
	m, err := Load(testmodels.Path(t, name), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ears := whisperEars(t)
	c := m.cfg.Talker
	type ranked struct {
		head        [2]int
		mean, worst float64
	}
	var all []ranked
	for layer := range c.Layers {
		for head := range c.Heads {
			m.align, m.aligned = [2]int{layer, head}, true
			mean, worst := voicedDistance(t, m, ears, voicedSentences[:1])
			all = append(all, ranked{m.align, mean, worst})
		}
	}
	slices.SortFunc(all, func(a, b ranked) int { return cmp.Compare(a.mean, b.mean) })
	t.Logf("median head: %.2f words away", all[len(all)/2].mean)
	for _, r := range all[:10] {
		m.align = r.head
		mean, worst := voicedDistance(t, m, ears, voicedSentences)
		t.Logf("layer %d, head %d: %.2f words away on one sentence; %.2f on all, %v at worst", r.head[0], r.head[1], r.mean, mean, worst)
	}
}

var voicedSentences = []struct{ voice, text string }{
	{"ryan", "The quick brown fox jumps over the lazy dog, and then it runs back home to sleep."},
	{"ryan", "Lisbon is the capital of Portugal, a city of hills, trams, and old yellow houses by the river."},
	{"serena", "I can remind you in thirty seconds, or whenever you like. Just tell me what to say."},
	{"ryan", "Sure! The meeting starts at three o'clock, so you still have about twenty minutes to prepare."},
	{"serena", "Photosynthesis turns sunlight, water, and carbon dioxide into sugar and oxygen inside green leaves."},
	{"ryan", "Well, that depends. If you want speed, take the train; if you want the view, drive along the coast."},
	{"serena", "Hello there. How are you doing today?"},
	{"ryan", "My favorite number is forty two, because it is the answer to life, the universe, and everything."},
}

func whisperEars(t *testing.T) *whisper.Transcriber {
	wm, err := whisper.Load(filepath.Join(testmodels.Path(t, testmodels.WhisperBaseEN)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { wm.Close() })
	ears, err := whisper.NewTranscriber(wm, whisper.LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ears.Close() })
	return ears
}

// voicedDistance speaks each sentence, streamed a word at a time, and
// returns how many words Voiced is from the word Whisper hears being
// spoken, on average over the frames of speech and at worst.
func voicedDistance(t *testing.T, m *Model, ears *whisper.Transcriber, sentences []struct{ voice, text string }) (mean, worst float64) {
	s, err := NewSynthesizer(m, LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var sum float64
	frames := 0
	resampler := speech.NewResampler()
	defer resampler.Close()
	for _, sen := range sentences {
		if err := s.Begin(context.Background(), speech.SpeakOptions{Voice: sen.voice, Language: speech.English}); err != nil {
			t.Fatal(err)
		}
		go func() {
			for _, w := range strings.SplitAfter(sen.text, " ") {
				s.Write([]byte(w))
			}
			s.End()
		}()
		var pcm []float32
		frame := make([]float32, FrameSamples)
		for {
			n, err := s.Read(frame)
			pcm = append(pcm, frame[:n]...)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		voiced := make([]int, len(pcm)/FrameSamples) // at each frame's middle
		for f := range voiced {
			voiced[f] = s.Voiced(f*FrameSamples + FrameSamples/2)
		}
		mono := make([]float32, len(pcm))
		n, err := resampler.Resample16kInto(pcm, SampleRate, 1, mono)
		if err != nil {
			t.Fatal(err)
		}
		var heard speech.Transcript
		if err := ears.Transcribe(context.Background(), mono[:n], speech.Options{Words: true}, &heard); err != nil {
			t.Fatal(err)
		}
		if len(heard.Words) == 0 {
			continue
		}
		starts := wordStarts(sen.text, &heard)
		words := wordSpans(sen.text)
		var errs []float64
		for f, v := range voiced {
			at := (float64(f) + 0.5) * FrameSamples / SampleRate
			if at < starts[0] || at > heard.Words[len(heard.Words)-1].End {
				continue // not speech
			}
			spoken := 0 // the word being spoken: the last one started
			for spoken+1 < len(starts) && starts[spoken+1] <= at {
				spoken++
			}
			reached := -1 // the word of the last byte voiced
			for reached+1 < len(words) && words[reached+1][0] < v {
				reached++
			}
			errs = append(errs, float64(reached-spoken))
		}
		for _, e := range errs {
			sum += math.Abs(e)
			worst = max(worst, math.Abs(e))
		}
		frames += len(errs)
	}
	return sum / float64(max(1, frames)), worst
}

// wordSpans returns the byte span of each word of text.
func wordSpans(text string) [][2]int {
	var spans [][2]int
	start := -1
	for i, r := range text + " " {
		switch {
		case unicode.IsSpace(r) && start >= 0:
			spans = append(spans, [2]int{start, i})
			start = -1
		case !unicode.IsSpace(r) && start < 0:
			start = i
		}
	}
	return spans
}

// wordStarts times each word of text by the words heard: an edit-distance
// alignment of the two word sequences, compared without case or
// punctuation, with a word heard differently ("thirty" as "30") timed
// between its neighbours.
func wordStarts(text string, heard *speech.Transcript) []float64 {
	norm := func(s string) string {
		return strings.Map(func(r rune) rune {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				return unicode.ToLower(r)
			}
			return -1
		}, s)
	}
	var a, b []string
	for _, w := range wordSpans(text) {
		a = append(a, norm(text[w[0]:w[1]]))
	}
	for _, w := range heard.Words {
		b = append(b, norm(string(heard.Text[w.TextStart:w.TextEnd])))
	}
	d := make([][]int, len(a)+1)
	for i := range d {
		d[i] = make([]int, len(b)+1)
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}
	for i := 1; i <= len(a); i++ {
		for j := 1; j <= len(b); j++ {
			sub := 1
			if a[i-1] == b[j-1] {
				sub = 0
			}
			d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+sub)
		}
	}
	starts := make([]float64, len(a))
	for i := range starts {
		starts[i] = math.NaN()
	}
	for i, j := len(a), len(b); i > 0 && j > 0; {
		sub := 1
		if a[i-1] == b[j-1] {
			sub = 0
		}
		switch d[i][j] {
		case d[i-1][j-1] + sub:
			starts[i-1] = heard.Words[j-1].Start
			i, j = i-1, j-1
		case d[i-1][j] + 1:
			i--
		default:
			j--
		}
	}
	// Interpolate the words heard differently.
	for i := range starts {
		if !math.IsNaN(starts[i]) {
			continue
		}
		lo, hi := i-1, i+1
		for hi < len(starts) && math.IsNaN(starts[hi]) {
			hi++
		}
		switch {
		case lo < 0 && hi == len(starts):
			starts[i] = 0
		case lo < 0:
			starts[i] = starts[hi]
		case hi == len(starts):
			starts[i] = starts[lo]
		default:
			starts[i] = starts[lo] + (starts[hi]-starts[lo])*float64(i-lo)/float64(hi-lo)
		}
	}
	return starts
}
