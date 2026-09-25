// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"math"
	"unicode"
	"unicode/utf8"
)

// TranscribeInto transcribes arbitrary-length mono 16 kHz PCM with the
// official full-file log-mel transform and successive mel windows.
// It uses deterministic English greedy decoding, carries previous text as
// decoder context, and applies Whisper's no-speech threshold. It writes into
// caller-owned storage. Repeating a call with the same input and sufficient
// output capacity allocates no heap objects after warmup.
//
// The current method returns plain text and uses Whisper's timestamp tokens
// to advance between windows. Temperature fallback and word timestamps are
// not part of this API.
func (t *Transcriber) TranscribeInto(pcm []float32, dst []byte) ([]byte, error) {
	text, _, _, err := t.transcribeFullInto(pcm, dst, nil, nil, 0)
	return text, err
}

// Segment records a retained Whisper segment. Times are seconds from the
// beginning of the recording. TextStart:TextEnd indexes the returned text.
type Segment struct {
	Start, End           float64
	TextStart, TextEnd   int
	WordStart, WordEnd   int
	tokenStart, tokenEnd int
}

// TranscribeSegmentsInto returns the same transcript as TranscribeInto plus
// model-derived 20 ms timestamp segments in caller-owned storage. Warm calls
// with sufficient text and segment capacity allocate no heap objects.
func (t *Transcriber) TranscribeSegmentsInto(pcm []float32, dst []byte, segments []Segment) ([]byte, []Segment, error) {
	text, timed, _, err := t.transcribeFullInto(pcm, dst, segments, nil, 1)
	return text, timed, err
}

// Word records forced-alignment timing for a word, with byte offsets into the
// returned transcript. Probability is the model's mean token probability.
type Word struct {
	Start, End         float64
	TextStart, TextEnd int
	Probability        float64
}

// TranscribeWordsInto adds word timing by aligning the selected tiny.en
// cross-attention heads using a second decoder pass over retained text tokens.
// Scratch is retained per Transcriber; warm calls reuse it.
func (t *Transcriber) TranscribeWordsInto(pcm []float32, dst []byte, segments []Segment, words []Word) ([]byte, []Segment, []Word, error) {
	return t.transcribeFullInto(pcm, dst, segments, words, 2)
}

func (t *Transcriber) transcribeFullInto(pcm []float32, dst []byte, segments []Segment, words []Word, mode uint8) ([]byte, []Segment, []Word, error) {
	if t == nil || t.closed {
		return dst, segments, words, ErrTranscriberClosed
	}
	frames, err := FullFeatureFrames(len(pcm))
	if err != nil {
		return dst, segments, words, err
	}
	need := frames * MelBins
	if cap(t.fullMel) < need {
		t.fullMel = make([]float32, need)
	}
	t.fullMel = t.fullMel[:need]
	if err := FullFeaturesInto(pcm, t.fullMel, t.fullFrontend); err != nil {
		return dst, segments, words, err
	}
	contentFrames := frames - MelFrames
	t.recordWords = mode == 2
	t.lastSpeechTimestamp = 0
	start := len(dst)
	segmentsStart, wordsStart := len(segments), len(words)
	t.history = t.history[:0]
	for seek := 0; seek < contentFrames; {
		segmentFrames := min(MelFrames, contentFrames-seek)
		firstSegment := len(segments)
		if mode == 2 {
			t.alignTokens = t.alignTokens[:0]
			t.alignOffsets = append(t.alignOffsets[:0], len(dst))
		}
		for mel := 0; mel < MelBins; mel++ {
			source := t.fullMel[mel*frames+seek : mel*frames+seek+segmentFrames]
			target := t.mel[mel*MelFrames : (mel+1)*MelFrames]
			copy(target, source)
			clear(target[segmentFrames:])
		}
		if err := t.model.EncodeInto(t.mel, t.audio, t.encoder); err != nil {
			return dst, segments, words, err
		}
		if err := t.model.BeginDecode(t.audio, t.decoder); err != nil {
			return dst, segments, words, err
		}
		promptLen, err := t.policy.PromptInto(t.tokens[:0], t.history, nil)
		if err != nil {
			return dst, segments, words, err
		}
		t.tokens = t.tokens[:promptLen]
		noSpeech := float64(0)
		for position, id := range t.tokens {
			if err := t.model.LogitsForTokenInto(id, position, t.decoder, t.logits); err != nil {
				return dst, segments, words, err
			}
			if id == t.tokenizer.SOT() {
				noSpeech = tokenProbability(t.logits, t.tokenizer.NoSpeech())
			}
		}
		logprob := float64(0)
		textEnd := promptLen
		for generated := 0; generated < TextContext/2 && len(t.tokens) <= TextContext; generated++ {
			next, err := t.policy.SelectNextInto(t.logits, t.logits, t.tokens)
			if err != nil {
				return dst, segments, words, err
			}
			// The silence rule below only consults average log probability
			// when the prompt's no-speech probability exceeds its threshold.
			if noSpeech > 0.6 {
				logprob += math.Log(tokenProbability(t.logits, next))
			}
			t.tokens = append(t.tokens, next)
			if next == t.tokenizer.EOT() {
				break
			}
			textEnd = len(t.tokens)
			if generated+1 >= TextContext/2 || len(t.tokens) > TextContext {
				break
			}
			if err := t.model.LogitsForTokenInto(next, len(t.tokens)-1, t.decoder, t.logits); err != nil {
				return dst, segments, words, err
			}
		}
		generatedTokens := t.tokens[promptLen:textEnd]
		avgLogprob := logprob / float64(len(generatedTokens)+1)
		if noSpeech > 0.6 && avgLogprob <= -1.0 {
			seek += segmentFrames
			continue
		}
		accepted, advance, pairBranch := timestampSeek(generatedTokens, t.tokenizer.TimestampBegin(), segmentFrames)
		windowSeek := seek
		seek += advance
		dst, segments, err = t.appendTranscribedSegmentsAt(dst, accepted, segmentFrames, pairBranch, windowSeek, segments, mode != 0)
		if err != nil {
			return dst, segments, words, err
		}
		if mode == 2 && len(t.alignTokens) != 0 {
			words, err = t.alignWordsForWindow(windowSeek, segmentFrames, dst, segments[firstSegment:], words)
			if err != nil {
				return dst, segments, words, err
			}
		}
	}
	invalidUTF8 := !utf8.Valid(dst[start:])
	oldLength := len(dst) - start
	repaired, err := t.repairTranscriptionUTF8(dst, start)
	if err == nil && invalidUTF8 {
		old := t.segmentText[:oldLength]
		for i := segmentsStart; i < len(segments); i++ {
			segments[i].TextStart = start + remapUTF8Boundary(old, segments[i].TextStart-start, false)
			segments[i].TextEnd = start + remapUTF8Boundary(old, segments[i].TextEnd-start, true)
		}
		for i := wordsStart; i < len(words); i++ {
			words[i].TextStart = start + remapUTF8Boundary(old, words[i].TextStart-start, false)
			words[i].TextEnd = start + remapUTF8Boundary(old, words[i].TextEnd-start, true)
		}
	}
	return repaired, segments, words, err
}

// appendTranscribedSegments mirrors transcribe.py's segment slicing and
// empty/zero-duration cleanup before tokens become future-window context.
func (t *Transcriber) appendTranscribedSegments(dst []byte, tokens []int, segmentFrames int, pairBranch bool) ([]byte, error) {
	text, _, err := t.appendTranscribedSegmentsAt(dst, tokens, segmentFrames, pairBranch, 0, nil, false)
	return text, err
}

func (t *Transcriber) appendTranscribedSegmentsAt(dst []byte, tokens []int, segmentFrames int, pairBranch bool, windowSeek int, segments []Segment, record bool) ([]byte, []Segment, error) {
	begin := t.tokenizer.TimestampBegin()
	start := 0
	hasPair := false
	for i := 1; i < len(tokens); i++ {
		if tokens[i-1] < begin || tokens[i] < begin {
			continue
		}
		hasPair = true
		var err error
		dst, segments, err = t.appendOneSegmentAt(dst, tokens[start:i], true, segmentFrames, windowSeek, segments, record)
		if err != nil {
			return dst, segments, err
		}
		start = i
	}
	if start < len(tokens) {
		return t.appendOneSegmentAt(dst, tokens[start:], hasPair || pairBranch, segmentFrames, windowSeek, segments, record)
	}
	return dst, segments, nil
}

func (t *Transcriber) appendOneSegmentAt(dst []byte, tokens []int, hasPair bool, segmentFrames, windowSeek int, segments []Segment, record bool) ([]byte, []Segment, error) {
	begin := t.tokenizer.TimestampBegin()
	startFrame, endFrame := windowSeek, windowSeek+segmentFrames
	if hasPair {
		// Each completed timestamp slice begins and ends with its boundaries.
		if len(tokens) > 1 && tokens[0] >= begin && tokens[len(tokens)-1] >= begin && tokens[0] == tokens[len(tokens)-1] {
			return dst, segments, nil
		}
		if len(tokens) == 0 {
			return dst, segments, nil
		}
		if tokens[0] >= begin {
			startFrame += (tokens[0] - begin) * 2
		}
		if tokens[len(tokens)-1] >= begin {
			endFrame = windowSeek + (tokens[len(tokens)-1]-begin)*2
		}
	} else {
		// Without a pair, transcribe.py uses the last timestamp as duration.
		for i := len(tokens) - 1; i >= 0; i-- {
			if tokens[i] >= begin {
				if tokens[i] != begin {
					segmentFrames = (tokens[i] - begin) * 2
				}
				break
			}
		}
		if segmentFrames == 0 {
			return dst, segments, nil
		}
		endFrame = windowSeek + segmentFrames
	}
	// Match transcribe.py's blank/instantaneous segment rule.
	if startFrame == endFrame {
		return dst, segments, nil
	}
	need := 0
	for _, id := range tokens {
		if id < 0 || id >= t.tokenizer.VocabSize() {
			return dst, segments, ErrTokenizerTokenRange
		}
		if id < t.tokenizer.EOT() {
			need += len(t.tokenizer.decoder[id])
		}
	}
	if cap(t.segmentText) < need {
		t.segmentText = make([]byte, 0, need)
	}
	t.segmentText = t.segmentText[:0]
	for _, id := range tokens {
		if id < t.tokenizer.EOT() {
			t.segmentText = append(t.segmentText, t.tokenizer.decoder[id]...)
		}
	}
	if blankWhisperText(t.segmentText) {
		return dst, segments, nil
	}
	offset := len(dst)
	decoded, err := t.tokenizer.DecodeInto(dst, tokens)
	if err != nil {
		return dst, segments, err
	}
	t.history = append(t.history, tokens...)
	if record {
		segment := Segment{Start: float64(startFrame) * 0.01, End: float64(endFrame) * 0.01, TextStart: offset, TextEnd: len(decoded)}
		if t.recordWords {
			segment.tokenStart = len(t.alignTokens)
			for _, id := range tokens {
				if id < t.tokenizer.EOT() {
					t.alignTokens = append(t.alignTokens, id)
					t.alignOffsets = append(t.alignOffsets, t.alignOffsets[len(t.alignOffsets)-1]+len(t.tokenizer.decoder[id]))
				}
			}
			segment.tokenEnd = len(t.alignTokens)
		}
		segments = append(segments, segment)
	}
	return decoded, segments, nil
}

func blankWhisperText(encoded []byte) bool {
	for len(encoded) != 0 {
		r, size := utf8.DecodeRune(encoded)
		if !whisperSpace(r) {
			return false
		}
		encoded = encoded[size:]
	}
	return true
}

func whisperSpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}

func trimWhisperSpace(raw []byte) []byte {
	for len(raw) != 0 {
		r, size := utf8.DecodeRune(raw)
		if !whisperSpace(r) {
			break
		}
		raw = raw[size:]
	}
	for len(raw) != 0 {
		r, size := utf8.DecodeLastRune(raw)
		if !whisperSpace(r) {
			break
		}
		raw = raw[:len(raw)-size]
	}
	return raw
}

// repairTranscriptionUTF8 implements tiktoken's UTF-8 errors="replace" after
// all retained segments have been concatenated. A token or segment can end
// midway through a codepoint that the next segment completes.
func (t *Transcriber) repairTranscriptionUTF8(dst []byte, start int) ([]byte, error) {
	raw := dst[start:]
	if utf8.Valid(raw) {
		return dst, nil
	}
	need := 0
	for i := 0; i < len(raw); {
		width, valid := utf8Prefix(raw[i:])
		if valid {
			need += width
		} else {
			need += 3
		}
		i += width
	}
	if start+need > cap(dst) {
		return dst[:start], ErrTranscriberTextCapacity
	}
	if cap(t.segmentText) < len(raw) {
		t.segmentText = make([]byte, len(raw))
	}
	copyRaw := t.segmentText[:len(raw)]
	copy(copyRaw, raw)
	dst = dst[:start+need]
	write := start
	for i := 0; i < len(copyRaw); {
		width, valid := utf8Prefix(copyRaw[i:])
		if valid {
			copy(dst[write:write+width], copyRaw[i:i+width])
			write += width
		} else {
			dst[write], dst[write+1], dst[write+2] = 0xef, 0xbf, 0xbd
			write += 3
		}
		i += width
	}
	return dst, nil
}

// utf8Prefix returns one valid sequence or the maximal prefix consumed by
// Python's UTF-8 replacement decoder before an invalid/missing continuation.
func utf8Prefix(src []byte) (int, bool) {
	first := src[0]
	if first < 0x80 {
		return 1, true
	}
	width, secondMin, secondMax := 0, byte(0x80), byte(0xbf)
	switch {
	case first >= 0xc2 && first <= 0xdf:
		width = 2
	case first == 0xe0:
		width, secondMin = 3, 0xa0
	case first == 0xed:
		width, secondMax = 3, 0x9f
	case first >= 0xe1 && first <= 0xef:
		width = 3
	case first == 0xf0:
		width, secondMin = 4, 0x90
	case first == 0xf4:
		width, secondMax = 4, 0x8f
	case first >= 0xf1 && first <= 0xf3:
		width = 4
	default:
		return 1, false
	}
	consumed := 1
	for consumed < width {
		if consumed >= len(src) {
			return consumed, false
		}
		x := src[consumed]
		if consumed == 1 {
			if x < secondMin || x > secondMax {
				return consumed, false
			}
		} else if x < 0x80 || x > 0xbf {
			return consumed, false
		}
		consumed++
	}
	return width, true
}

// timestampSeek follows the token slicing and seek rule of transcribe.py.
// A final incomplete timestamp segment is decoded again in the next window.
func timestampSeek(tokens []int, timestampBegin, segmentFrames int) ([]int, int, bool) {
	lastConsecutive := -1
	for i := 1; i < len(tokens); i++ {
		if tokens[i-1] >= timestampBegin && tokens[i] >= timestampBegin {
			lastConsecutive = i
		}
	}
	if lastConsecutive < 0 {
		return tokens, segmentFrames, false
	}
	last := len(tokens) - 1
	singleTimestampEnding := last >= 1 && tokens[last] >= timestampBegin && tokens[last-1] < timestampBegin
	if singleTimestampEnding {
		return tokens, segmentFrames, true
	}
	lastTimestamp := tokens[lastConsecutive-1] - timestampBegin
	advance := lastTimestamp * 2 // two mel frames per 20 ms timestamp tick
	if advance < 1 {
		// Guard against a zero-progress model output. The pinned Python
		// loop would revisit the same window indefinitely in this case.
		advance = 1
	}
	return tokens[:lastConsecutive], advance, true
}

func tokenProbability(logits []float32, id int) float64 {
	maximum := float64(math.Inf(-1))
	for _, x := range logits {
		if float64(x) > maximum {
			maximum = float64(x)
		}
	}
	if math.IsInf(maximum, -1) {
		return 0
	}
	sum := float64(0)
	for _, x := range logits {
		sum += math.Exp(float64(x) - maximum)
	}
	return math.Exp(float64(logits[id])-maximum) / sum
}

// remapUTF8Boundary maps a byte boundary in the generated token stream into
// the repaired UTF-8 transcript. A boundary inside a multibyte code point is
// expanded outward so exposed slices remain valid UTF-8.
func remapUTF8Boundary(raw []byte, boundary int, end bool) int {
	if boundary <= 0 {
		return 0
	}
	oldAt, newAt := 0, 0
	for oldAt < len(raw) {
		width, valid := utf8Prefix(raw[oldAt:])
		outputWidth := width
		if !valid {
			outputWidth = 3
		}
		if boundary <= oldAt+width {
			if boundary < oldAt+width && !end {
				return newAt
			}
			return newAt + outputWidth
		}
		oldAt += width
		newAt += outputWidth
	}
	return newAt
}
