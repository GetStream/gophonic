// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"errors"
	"math"
	"sort"
)

var ErrWordAlignmentContext = errors.New("whisper: word alignment exceeds text context")
var ErrWordAlignmentModel = errors.New("whisper: word alignment heads are unavailable for this model")

func (t *Transcriber) alignWordsForWindow(windowSeek, segmentFrames int, text []byte, segments []Segment, words []Word) ([]Word, error) {
	heads := alignmentHeadsForDims(t.model.dims)
	if len(heads) == 0 {
		return words, ErrWordAlignmentModel
	}
	tokenCount := len(t.alignTokens)
	positions := tokenCount + 3 // SOT, NoTimestamps, text, EOT
	if positions > TextContext {
		return words, ErrWordAlignmentContext
	}
	audioFrames := min(AudioFrames, segmentFrames/2)
	if audioFrames == 0 {
		return words, nil
	}
	captureSize := len(heads) * positions * audioFrames
	if cap(t.alignCapture.probabilities) < captureSize {
		t.alignCapture.probabilities = make([]float32, captureSize)
	}
	t.alignCapture.probabilities = t.alignCapture.probabilities[:captureSize]
	t.alignCapture.frames = audioFrames
	t.alignCapture.positions = positions
	t.alignCapture.heads = heads
	if cap(t.alignProbabilities) < tokenCount {
		t.alignProbabilities = make([]float64, tokenCount)
	}
	t.alignProbabilities = t.alignProbabilities[:tokenCount]
	t.decoder.nextPos = 0 // Cross-KV remains valid; each self-KV row is overwritten before reuse.
	t.decoder.alignment = &t.alignCapture
	for pos := 0; pos < positions; pos++ {
		id := t.tokenizer.EOT()
		switch {
		case pos == 0:
			id = t.tokenizer.SOT()
		case pos == 1:
			id = t.tokenizer.NoTimestamps()
		case pos < positions-1:
			id = t.alignTokens[pos-2]
		}
		if err := t.model.decodeTokenInto(id, pos, t.decoder, t.logits, pos >= 1 && pos <= tokenCount); err != nil {
			t.decoder.alignment = nil
			return words, err
		}
		if pos >= 1 && pos <= tokenCount {
			t.alignProbabilities[pos-1] = textTokenProbability(t.logits, t.alignTokens[pos-1], t.tokenizer.EOT())
		}
	}
	t.decoder.alignment = nil
	rows := positions - 2 // NoTimestamps plus one row per text token.
	matrixSize := rows * audioFrames
	if cap(t.alignMatrix) < matrixSize {
		t.alignMatrix = make([]float32, matrixSize)
	}
	t.alignMatrix = t.alignMatrix[:matrixSize]
	normalizeAlignmentInto(t.alignMatrix, &t.alignCapture, positions)
	jumps := t.alignmentJumps(rows, audioFrames)
	offset := float64(windowSeek) * 0.01
	for si := range segments {
		seg := &segments[si]
		seg.WordStart = len(words)
		for begin := seg.tokenStart; begin < seg.tokenEnd; {
			end := begin + 1
			for end < seg.tokenEnd && !alignmentWordBreak(text[t.alignOffsets[end]:t.alignOffsets[end+1]]) {
				end++
			}
			startTime := math.Round((offset+float64(jumps[begin])*0.02)*100) / 100
			endTime := math.Round((offset+float64(jumps[end])*0.02)*100) / 100
			if endTime < startTime {
				endTime = startTime
			}
			var probability float64
			for _, value := range t.alignProbabilities[begin:end] {
				probability += value
			}
			probability /= float64(end - begin)
			word := Word{Start: startTime, End: endTime, TextStart: t.alignOffsets[begin], TextEnd: t.alignOffsets[end], Probability: probability}
			piece := text[word.TextStart:word.TextEnd]
			if len(words) > seg.WordStart && alignmentTrailingPunctuation(piece) {
				previous := &words[len(words)-1]
				previous.TextEnd = word.TextEnd
				previous.End = word.End
				// The reference keeps the previous word probability when punctuation is merged.
			} else {
				words = append(words, word)
			}
			begin = end
		}
		seg.WordEnd = len(words)
	}
	t.refineWordBoundaries(text, segments, words)
	return words, nil
}

func alignmentWordBreak(piece []byte) bool {
	if len(piece) == 0 {
		return false
	}
	return piece[0] == ' ' || piece[0] == '\n' || piece[0] == '\t' || alignmentPunctuation(piece[0])
}

func alignmentPunctuation(b byte) bool {
	switch b {
	case '.', ',', '!', '?', ';', ':', ')', ']', '}', '"', '\'', '-', '(':
		return true
	}
	return false
}

func alignmentTrailingPunctuation(piece []byte) bool {
	if len(piece) == 0 || len(piece) > 3 {
		return false
	}
	for _, b := range piece {
		if !alignmentPunctuation(b) {
			return false
		}
	}
	return true
}

// alignmentJumps runs monotone dynamic time warping against the negative
// attention matrix and returns the first audio frame for each token row.
func (t *Transcriber) alignmentJumps(rows, frames int) []int {
	stride := frames + 1
	traceSize := (rows + 1) * stride
	if cap(t.alignTrace) < traceSize {
		t.alignTrace = make([]byte, traceSize)
	}
	trace := t.alignTrace[:traceSize]
	if cap(t.alignCostPrevious) < stride {
		t.alignCostPrevious = make([]float64, stride)
	}
	if cap(t.alignCostCurrent) < stride {
		t.alignCostCurrent = make([]float64, stride)
	}
	previous, current := t.alignCostPrevious[:stride], t.alignCostCurrent[:stride]
	previous[0] = 0
	for j := 1; j < stride; j++ {
		previous[j] = math.Inf(1)
	}
	for i := 1; i <= rows; i++ {
		current[0] = math.Inf(1)
		for j := 1; j <= frames; j++ {
			diagonal, up, left := previous[j-1], previous[j], current[j-1]
			direction, best := byte(2), left
			if diagonal < up && diagonal < left {
				direction, best = 0, diagonal
			} else if up < diagonal && up < left {
				direction, best = 1, up
			}
			current[j] = -float64(t.alignMatrix[(i-1)*frames+j-1]) + best
			trace[i*stride+j] = direction
		}
		previous, current = current, previous
	}
	t.alignCostPrevious, t.alignCostCurrent = previous, current
	if cap(t.alignJumps) < rows {
		t.alignJumps = make([]int, rows)
	}
	jumps := t.alignJumps[:rows]
	for i := range jumps {
		jumps[i] = 0
	}
	for i, j := rows, frames; i > 0 || j > 0; {
		if i > 0 && j > 0 {
			jumps[i-1] = j - 1
		}
		switch {
		case i == 0:
			j--
		case j == 0:
			i--
		default:
			switch trace[i*stride+j] {
			case 0:
				i--
				j--
			case 1:
				i--
			default:
				j--
			}
		}
	}
	return jumps
}

// textTokenProbability matches find_alignment's softmax over ordinary text IDs.
func textTokenProbability(logits []float32, id, eot int) float64 {
	if id < 0 || id >= eot {
		return 0
	}
	maximum := float64(math.Inf(-1))
	for _, value := range logits[:eot] {
		if float64(value) > maximum {
			maximum = float64(value)
		}
	}
	var sum float64
	for _, value := range logits[:eot] {
		sum += math.Exp(float64(value) - maximum)
	}
	return math.Exp(float64(logits[id])-maximum) / sum
}

// refineWordBoundaries follows Whisper's median-duration and segment-edge
// adjustments after attention alignment. Source: OpenAI Whisper commit
// 86098128c0b4f24f0e2aa2994de830614b474227, whisper/timing.py.
func (t *Transcriber) refineWordBoundaries(text []byte, segments []Segment, words []Word) {
	first := len(words)
	for _, segment := range segments {
		if segment.WordStart < first {
			first = segment.WordStart
		}
	}
	if first >= len(words) {
		return
	}
	if cap(t.alignDurations) < len(words)-first {
		t.alignDurations = make([]float64, len(words)-first)
	}
	durations := t.alignDurations[:0]
	for _, word := range words[first:] {
		if word.End > word.Start {
			durations = append(durations, word.End-word.Start)
		}
	}
	median := 0.0
	if len(durations) != 0 {
		sort.Float64s(durations)
		middle := len(durations) / 2
		median = durations[middle]
		if len(durations)%2 == 0 {
			median = (durations[middle-1] + durations[middle]) * 0.5
		}
		median = math.Min(0.7, median)
	}
	maximum := 2 * median
	if maximum > 0 {
		for i := first; i < len(words); i++ {
			if words[i].End-words[i].Start <= maximum {
				continue
			}
			piece := text[words[i].TextStart:words[i].TextEnd]
			if len(piece) != 0 && (piece[len(piece)-1] == '.' || piece[len(piece)-1] == '!' || piece[len(piece)-1] == '?') {
				words[i].End = words[i].Start + maximum
			} else if i > first {
				before := text[words[i-1].TextStart:words[i-1].TextEnd]
				if len(before) != 0 && (before[len(before)-1] == '.' || before[len(before)-1] == '!' || before[len(before)-1] == '?') {
					words[i].Start = words[i].End - maximum
				}
			}
		}
	}
	for i := range segments {
		segment := &segments[i]
		if segment.WordStart == segment.WordEnd {
			continue
		}
		firstWord := &words[segment.WordStart]
		lastWord := &words[segment.WordEnd-1]
		if maximum > 0 && firstWord.End-t.lastSpeechTimestamp > median*4 &&
			(firstWord.End-firstWord.Start > maximum || (segment.WordEnd-segment.WordStart > 1 && words[segment.WordStart+1].End-firstWord.Start > maximum*2)) {
			if segment.WordEnd-segment.WordStart > 1 {
				second := &words[segment.WordStart+1]
				if second.End-second.Start > maximum {
					boundary := math.Max(second.End/2, second.End-maximum)
					firstWord.End, second.Start = boundary, boundary
				}
			}
			firstWord.Start = math.Max(0, firstWord.End-maximum)
		}
		if segment.Start < firstWord.End && segment.Start-0.5 > firstWord.Start {
			firstWord.Start = math.Max(0, math.Min(firstWord.End-median, segment.Start))
		} else {
			segment.Start = firstWord.Start
		}
		if segment.End > lastWord.Start && segment.End+0.5 < lastWord.End {
			lastWord.End = math.Max(lastWord.Start+median, segment.End)
		} else {
			segment.End = lastWord.End
		}
		t.lastSpeechTimestamp = segment.End
	}
}
