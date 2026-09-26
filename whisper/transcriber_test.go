// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/GetStream/gophonic/internal/testmodels"
	"github.com/thesyncim/vibejson"
)

func TestTranscriberOfficialJFK(t *testing.T) {
	path := testmodels.Path(t, testmodels.WhisperTinyEN)
	m, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := NewTranscriber(m, LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	manifest, err := os.ReadFile(filepath.Join("..", "testdata", "whisper", "jfk.oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Transcript string `json:"transcript"`
	}
	if err := vibejson.Unmarshal(manifest, &oracle); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 0, 2048)
	got, err := worker.TranscribeWindowInto(pcm, buf)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != oracle.Transcript {
		t.Fatalf("transcript %q, want %q", got, oracle.Transcript)
	}
	full, err := worker.TranscribeInto(pcm, buf)
	if err != nil {
		t.Fatal(err)
	}
	fullOracleData, err := os.ReadFile(filepath.Join("..", "testdata", "whisper", "fullfile.greedy_notimestamps.oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fullOracle struct {
		SourceCommit     string `json:"source_commit"`
		CheckpointSHA256 string `json:"checkpoint_sha256"`
		Cases            []struct {
			Name       string `json:"name"`
			PCMSHA256  string `json:"pcm_sha256"`
			Transcript string `json:"transcript"`
		} `json:"cases"`
	}
	if err := vibejson.Unmarshal(fullOracleData, &fullOracle); err != nil {
		t.Fatal(err)
	}
	if fullOracle.SourceCommit != "86098128c0b4f24f0e2aa2994de830614b474227" ||
		fullOracle.CheckpointSHA256 != "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03" {
		t.Fatal("full-file oracle source or checkpoint pin changed")
	}
	if len(fullOracle.Cases) != 3 {
		t.Fatal("full-file oracle must include JFK, silence, and long audio")
	}
	cases := make(map[string]struct{ transcript, pcmSHA string }, len(fullOracle.Cases))
	for _, c := range fullOracle.Cases {
		cases[c.Name] = struct{ transcript, pcmSHA string }{c.Transcript, c.PCMSHA256}
	}
	if len(cases) != 3 {
		t.Fatal("full-file oracle contains duplicate or missing case names")
	}
	jfkSHA := sha256.Sum256(data)
	if cases["jfk"].pcmSHA != hex.EncodeToString(jfkSHA[:]) {
		t.Fatal("full-file JFK PCM pin differs")
	}
	if string(full) != cases["jfk"].transcript {
		t.Fatalf("full-file transcript %q, want %q", full, cases["jfk"].transcript)
	}
	silence := make([]float32, 16000)
	quiet, err := worker.TranscribeInto(silence, buf)
	if err != nil {
		t.Fatal(err)
	}
	zeroBytes := make([]byte, 16000*4)
	zeroSHA := sha256.Sum256(zeroBytes)
	if cases["silence_1s"].pcmSHA != hex.EncodeToString(zeroSHA[:]) {
		t.Fatal("full-file silence PCM pin differs")
	}
	if string(quiet) != cases["silence_1s"].transcript {
		t.Fatalf("silence transcript %q, want %q", quiet, cases["silence_1s"].transcript)
	}
	if allocs := testing.AllocsPerRun(2, func() {
		if _, err := worker.TranscribeInto(silence, buf); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warm full-file allocations = %v", allocs)
	}
	longPCM := make([]float32, len(pcm)+35*16000)
	copy(longPCM, pcm)
	hash := sha256.New()
	_, _ = hash.Write(data)
	for i := 0; i < 35; i++ {
		_, _ = hash.Write(zeroBytes)
	}
	if cases["jfk_plus_35s_silence"].pcmSHA != hex.EncodeToString(hash.Sum(nil)) {
		t.Fatal("full-file long PCM pin differs")
	}
	longText, err := worker.TranscribeInto(longPCM, buf)
	if err != nil {
		t.Fatal(err)
	}
	if string(longText) != cases["jfk_plus_35s_silence"].transcript {
		t.Fatalf("long transcript %q, want %q", longText, cases["jfk_plus_35s_silence"].transcript)
	}
}

func TestTimestampSeek(t *testing.T) {
	const start = 50364
	tests := []struct {
		name             string
		tokens           []int
		wantLen, advance int
	}{
		{"no timestamps", []int{100, 200}, 2, 3000},
		{"closed pair", []int{start, 100, start + 50, start + 50, 200, start + 90}, 6, 3000},
		{"unfinished pair", []int{start, 100, start + 50, start + 50, 200}, 3, 100},
		{"zero seek progress", []int{start, start, 200}, 1, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, advance, _ := timestampSeek(tt.tokens, start, 3000)
			if len(got) != tt.wantLen || advance != tt.advance {
				t.Fatalf("len=%d seek=%d, want len=%d seek=%d", len(got), advance, tt.wantLen, tt.advance)
			}
		})
	}
}

func TestFullFileClearsZeroDurationSegment(t *testing.T) {
	if !blankWhisperText([]byte("\u2003")) || !blankWhisperText([]byte{0x1c, 0x1d, 0x1e, 0x1f}) || blankWhisperText([]byte(" A")) {
		t.Fatal("Whisper segment blank-text classification differs")
	}
	tokenizer, err := newTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	a, err := tokenizer.EncodeInto(make([]int, 0, 8), " A")
	if err != nil {
		t.Fatal(err)
	}
	b, err := tokenizer.EncodeInto(make([]int, 0, 8), " B")
	if err != nil {
		t.Fatal(err)
	}
	begin := tokenizer.TimestampBegin()
	ids := make([]int, 0, len(a)+len(b)+4)
	ids = append(ids, begin)
	ids = append(ids, a...)
	ids = append(ids, begin, begin)
	ids = append(ids, b...)
	ids = append(ids, begin+10)
	worker := &Transcriber{tokenizer: tokenizer, history: make([]int, 0, 32)}
	got, err := worker.appendTranscribedSegments(make([]byte, 0, 128), ids, MelFrames, true)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != " B" {
		t.Fatalf("retained text %q, want second segment only", got)
	}
	if len(worker.history) != len(b)+2 || worker.history[0] != begin || worker.history[len(worker.history)-1] != begin+10 {
		t.Fatalf("retained prompt tokens %v", worker.history)
	}
	worker.history = worker.history[:0]
	unfinished := make([]int, 0, len(a)+4)
	unfinished = append(unfinished, begin+50)
	unfinished = append(unfinished, a...)
	unfinished = append(unfinished, begin+50, begin+50)
	unfinished = append(unfinished, b...)
	accepted, _, pairBranch := timestampSeek(unfinished, begin, MelFrames)
	got, err = worker.appendTranscribedSegments(got[:0], accepted, MelFrames, pairBranch)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 || len(worker.history) != 0 {
		t.Fatalf("instantaneous unfinished segment leaked text=%q tokens=%v", got, worker.history)
	}
	space, err := tokenizer.EncodeInto(make([]int, 0, 8), " ")
	if err != nil {
		t.Fatal(err)
	}
	blank, err := worker.appendTranscribedSegments(nil, space, MelFrames, false)
	if err != nil || len(blank) != 0 {
		t.Fatalf("blank segment with zero output capacity returned %q, %v", blank, err)
	}
}

func TestTranscriberUTF8Replacement(t *testing.T) {
	if got := string(trimWhisperSpace([]byte("\x1c\u2003 A \u2003\x1f"))); got != "A" {
		t.Fatalf("Python-compatible whitespace trim = %q", got)
	}
	tests := []struct {
		input []byte
		want  string
	}{
		{[]byte{0xc3}, "�"},
		{[]byte{0xe2, 0x82}, "�"},
		{[]byte{0xc3, '('}, "�("},
		{[]byte{0xe0, 0x80, 0x80}, "���"},
		{[]byte{0xe2, 0x82, 0xff}, "��"},
		{[]byte{0xc3, 0xa9}, "é"},
	}
	worker := &Transcriber{segmentText: make([]byte, 0, 32)}
	for _, tt := range tests {
		buf := make([]byte, len(tt.input), 32)
		copy(buf, tt.input)
		got, err := worker.repairTranscriptionUTF8(buf, 0)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != tt.want {
			t.Fatalf("repair %x = %q, want %q", tt.input, got, tt.want)
		}
	}
	buf := []byte{0xc3}
	if _, err := worker.repairTranscriptionUTF8(buf, 0); err != ErrTranscriberTextCapacity {
		t.Fatalf("short repair capacity error = %v", err)
	}
}

func TestTranscriberRejectsClosedAndLongWindow(t *testing.T) {
	if _, err := NewTranscriber(nil, LaneOptions{}); err == nil {
		t.Fatal("accepted nil model")
	}
	worker, err := NewTranscriber(&Model{}, LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := worker.TranscribeWindowInto(make([]float32, WindowSamples+1), nil); err != ErrTranscriberWindow {
		t.Fatalf("long window error = %v", err)
	}
	worker.Close()
	worker.Close()
	if _, err := worker.TranscribeWindowInto(nil, nil); err != ErrTranscriberClosed {
		t.Fatalf("closed worker error = %v", err)
	}
}

func TestTranscriberExplicitWorkerBudget(t *testing.T) {
	for _, workers := range []int{-1, 65} {
		if _, err := NewTranscriber(&Model{}, LaneOptions{Threads: workers}); err == nil {
			t.Fatalf("accepted %d workers", workers)
		}
	}
	worker, err := NewTranscriber(&Model{}, LaneOptions{Threads: 12})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	if got := worker.encoder.attention.workers; got != 12 {
		t.Fatalf("encoder has %d workers, want 12", got)
	}
	if worker.decoder.gemm != worker.encoder.gemm {
		t.Fatal("decoder does not share the configured encoder executor")
	}
}

func TestSegmentTimestampOffsets(t *testing.T) {
	tok, err := newTokenizer(EnglishOnly)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.EncodeInto(make([]int, 0, 8), " hello")
	if err != nil {
		t.Fatal(err)
	}
	begin := tok.TimestampBegin()
	tokens := append([]int{begin + 25}, ids...)
	tokens = append(tokens, begin+75)
	worker := &Transcriber{tokenizer: tok, history: make([]int, 0, 16)}
	text, segments, err := worker.appendTranscribedSegmentsAt(make([]byte, 0, 64), tokens, MelFrames, true, 100, make([]Segment, 0, 2), true)
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) != 1 || segments[0].Start != 1.5 || segments[0].End != 2.5 || string(text[segments[0].TextStart:segments[0].TextEnd]) != " hello" {
		t.Fatalf("segments=%+v text=%q", segments, text)
	}
}

func TestOfficialJFKSegmentTimestamps(t *testing.T) {
	path := testmodels.Path(t, testmodels.WhisperTinyEN)
	model, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := NewTranscriber(model, LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	text, segments, err := worker.TranscribeSegmentsInto(pcm, make([]byte, 0, 4096), make([]Segment, 0, 64))
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) != 1 || segments[0].Start != 0 || segments[0].End != 11 || string(text[segments[0].TextStart:segments[0].TextEnd]) != " And so my fellow Americans ask not what your country can do for you ask what you can do for your country." {
		t.Fatalf("segments=%+v text=%q", segments, text)
	}
}

func TestOfficialJFKWordTimestamps(t *testing.T) {
	path := testmodels.Path(t, testmodels.WhisperTinyEN)
	model, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := NewTranscriber(model, LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	text, segments, words, err := worker.TranscribeWordsInto(pcm, make([]byte, 0, 4096), make([]Segment, 0, 64), make([]Word, 0, 128))
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) != 1 || len(words) < 10 || segments[0].WordStart != 0 || segments[0].WordEnd != len(words) {
		t.Fatalf("segments=%+v words=%+v", segments, words)
	}
	for i, word := range words {
		if word.Start < 0 || word.End < word.Start || word.End > 11.1 || word.TextStart < 0 || word.TextEnd > len(text) || word.TextStart >= word.TextEnd {
			t.Fatalf("invalid word %d: %+v", i, word)
		}
		t.Logf("%q %.2f %.2f p=%.3f", text[word.TextStart:word.TextEnd], word.Start, word.End, word.Probability)
	}
}

func TestOfficialJFKWordTimestampsWarmZeroAlloc(t *testing.T) {
	path := testmodels.Path(t, testmodels.WhisperTinyEN)
	model, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := NewTranscriber(model, LaneOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	text := make([]byte, 0, 4096)
	segments := make([]Segment, 0, 64)
	words := make([]Word, 0, 128)
	call := func() {
		transcript, timed, aligned, err := worker.TranscribeWordsInto(pcm, text[:0], segments[:0], words[:0])
		if err != nil || len(transcript) == 0 || len(timed) == 0 || len(aligned) == 0 {
			panic("alignment failed")
		}
	}
	call()
	allocs := testing.AllocsPerRun(3, call)
	if allocs != 0 {
		t.Fatalf("warmed word alignment: %g allocs/op", allocs)
	}
}

func TestEnglishModelWordAlignmentHeads(t *testing.T) {
	for _, tc := range []struct{ state, layers, heads, count int }{{384, 4, 6, 8}, {512, 6, 8, 5}, {768, 12, 12, 19}, {1024, 24, 16, 18}} {
		selected := alignmentHeadsForDims(modelDims{TextState: tc.state, TextLayers: tc.layers, TextHeads: tc.heads})
		if len(selected) != tc.count {
			t.Fatalf("%dx%d heads=%d, want %d", tc.layers, tc.heads, len(selected), tc.count)
		}
		for _, head := range selected {
			if head.layer >= tc.layers || head.head >= tc.heads {
				t.Fatalf("invalid alignment head %+v", head)
			}
		}
	}
}

func TestOfficialOtherEnglishModelWordTimestamps(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "testdata", "whisper_jfk.pcm.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, len(data)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	for _, name := range []string{testmodels.WhisperBaseEN, testmodels.WhisperSmallEN} {
		t.Run(name, func(t *testing.T) {
			path := testmodels.Path(t, name)
			model, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			worker, err := NewTranscriber(model, LaneOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer worker.Close()
			text, segments, words, err := worker.TranscribeWordsInto(pcm, make([]byte, 0, 4096), make([]Segment, 0, 64), make([]Word, 0, 128))
			if err != nil || len(text) == 0 || len(segments) == 0 || len(words) < 10 {
				t.Fatalf("text=%q segments=%d words=%d err=%v", text, len(segments), len(words), err)
			}
			t.Logf("%s: %d words, first=%q %.2f-%.2f", name, len(words), text[words[0].TextStart:words[0].TextEnd], words[0].Start, words[0].End)
		})
	}
}
