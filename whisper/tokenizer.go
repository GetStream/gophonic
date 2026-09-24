// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package whisper

import (
	"bufio"
	"bytes"
	"embed"
	"encoding/base64"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// EnglishOnly selects the GPT-2 byte-pair vocabulary used by tiny.en.
// Multilingual selects Whisper's multilingual vocabulary.
type TokenizerKind uint8

const (
	EnglishOnly TokenizerKind = iota
	Multilingual
)

var (
	ErrTokenizerKind           = errors.New("whisper: unsupported tokenizer kind")
	ErrTokenizerOutputTooSmall = errors.New("whisper: tokenizer output buffer is too small")
	ErrTokenizerTokenRange     = errors.New("whisper: token ID is outside the tokenizer vocabulary")
	ErrTokenizerSpecialToken   = errors.New("whisper: input contains a special token")
	ErrTokenizerNil            = errors.New("whisper: nil tokenizer")
)

// Whisper's public source list contains 100 languages. The reference
// get_tokenizer defaults to 99, which is also the count embedded in tiny.en.
var whisperLanguageCodes = [...]string{
	"en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
	"pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
	"he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
	"th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
	"te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
	"br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
	"gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
	"ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
	"ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
	"mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
}

//go:embed assets/gpt2.tiktoken assets/multilingual.tiktoken
var tokenizerAssets embed.FS

const whisperLanguageCount = 99
const whisperTimestampCount = 1501

// Tokenizer is an immutable byte-level BPE tokenizer. Its maps are built once
// by NewTokenizer, so independent goroutines can encode and decode concurrently
// when each call has its own destination buffer.
type Tokenizer struct {
	kind       TokenizerKind
	vocabSize  int
	baseSize   int
	encoder    map[string]int
	decoder    [][]byte
	merges     map[uint64]int
	specialIDs map[string]int

	eot            int
	sot            int
	translate      int
	transcribe     int
	sotLM          int
	sotPrev        int
	noSpeech       int
	noTimestamps   int
	timestampBegin int
	languageTokens [whisperLanguageCount]int
}

// NewTokenizer loads the pinned, bundled Whisper vocabulary and compiles its
// BPE merge rules. EnglishOnly is the 51864-token tiny.en tokenizer; Multilingual
// includes the extra multilingual base token and has 51865 tokens.
func NewTokenizer(kind TokenizerKind) (*Tokenizer, error) {
	var asset string
	switch kind {
	case EnglishOnly:
		asset = "assets/gpt2.tiktoken"
	case Multilingual:
		asset = "assets/multilingual.tiktoken"
	default:
		return nil, ErrTokenizerKind
	}
	data, err := tokenizerAssets.ReadFile(asset)
	if err != nil {
		return nil, fmt.Errorf("read bundled Whisper vocabulary: %w", err)
	}
	t := &Tokenizer{
		kind:       kind,
		encoder:    make(map[string]int, 50257),
		specialIDs: make(map[string]int, 1608),
	}
	if err := t.readRanks(data); err != nil {
		return nil, err
	}
	t.baseSize = len(t.encoder)
	if err := t.addSpecialTokens(); err != nil {
		return nil, err
	}
	t.compileMerges()
	return t, nil
}

func (t *Tokenizer) readRanks(data []byte) error {
	s := bufio.NewScanner(bytes.NewReader(data))
	// A handful of vocabulary entries are longer than Scanner's default token
	// size; the pinned assets remain comfortably below this explicit limit.
	s.Buffer(make([]byte, 4096), 1<<20)
	maxRank := -1
	for s.Scan() {
		line := s.Text()
		space := strings.IndexByte(line, ' ')
		if space <= 0 || space == len(line)-1 {
			return errors.New("whisper: malformed tiktoken rank line")
		}
		var payload []byte
		var decodeErr error
		if line[:space] == "=" {
			// The pinned multilingual rank file includes one empty byte token
			// at rank 50256. Upstream base64.b64decode accepts this spelling.
			payload = []byte{}
		} else {
			payload, decodeErr = base64.StdEncoding.DecodeString(line[:space])
		}
		if decodeErr != nil {
			return errors.New("whisper: malformed tiktoken byte token")
		}
		rank, err := strconv.Atoi(line[space+1:])
		if err != nil || rank < 0 {
			return errors.New("whisper: malformed tiktoken rank")
		}
		key := string(payload)
		if _, exists := t.encoder[key]; exists {
			return errors.New("whisper: duplicate tiktoken byte token")
		}
		t.encoder[key] = rank
		if rank > maxRank {
			maxRank = rank
		}
	}
	if err := s.Err(); err != nil {
		return err
	}
	if len(t.encoder) == 0 || maxRank+1 != len(t.encoder) {
		return errors.New("whisper: tiktoken ranks are not contiguous")
	}
	t.baseSize = len(t.encoder)
	t.decoder = make([][]byte, t.baseSize)
	seen := make([]bool, t.baseSize)
	for token, rank := range t.encoder {
		if rank >= t.baseSize || seen[rank] {
			return errors.New("whisper: invalid or duplicate tiktoken rank")
		}
		seen[rank] = true
		t.decoder[rank] = []byte(token)
	}
	for _, ok := range seen {
		if !ok {
			return errors.New("whisper: tiktoken ranks are not contiguous")
		}
	}
	return nil
}

func (t *Tokenizer) addSpecialTokens() error {
	id := t.baseSize
	add := func(name string) int {
		t.specialIDs[name] = id
		t.decoder = append(t.decoder, []byte(name))
		id++
		return id - 1
	}
	t.eot = add("<|endoftext|>")
	t.sot = add("<|startoftranscript|>")
	for i := 0; i < whisperLanguageCount; i++ {
		t.languageTokens[i] = add("<|" + whisperLanguageCodes[i] + "|>")
	}
	t.translate = add("<|translate|>")
	t.transcribe = add("<|transcribe|>")
	t.sotLM = add("<|startoflm|>")
	t.sotPrev = add("<|startofprev|>")
	t.noSpeech = add("<|nospeech|>")
	t.noTimestamps = add("<|notimestamps|>")
	t.timestampBegin = id
	for i := 0; i < whisperTimestampCount; i++ {
		name := "<|" + strconv.FormatFloat(float64(i)*0.02, 'f', 2, 64) + "|>"
		add(name)
	}
	t.vocabSize = id
	if t.vocabSize != t.baseSize+1608 {
		return errors.New("whisper: unexpected special token count")
	}
	return nil
}

func (t *Tokenizer) compileMerges() {
	t.merges = make(map[uint64]int, len(t.encoder)*3)
	for payload, merged := range t.encoder {
		for split := 1; split < len(payload); split++ {
			left, leftOK := t.encoder[payload[:split]]
			if !leftOK {
				continue
			}
			right, rightOK := t.encoder[payload[split:]]
			if rightOK {
				t.merges[mergeKey(left, right)] = merged
			}
		}
	}
}

func mergeKey(left, right int) uint64 {
	return uint64(uint32(left))<<32 | uint64(uint32(right))
}

// Kind returns which bundled vocabulary this tokenizer uses.
func (t *Tokenizer) Kind() TokenizerKind { return t.kind }

// VocabSize returns the number of valid token IDs.
func (t *Tokenizer) VocabSize() int { return t.vocabSize }

// EOT is the end-of-text token ID.
func (t *Tokenizer) EOT() int { return t.eot }

// SOT is the start-of-transcript token ID.
func (t *Tokenizer) SOT() int { return t.sot }

// Translate is the translate-task token ID.
func (t *Tokenizer) Translate() int { return t.translate }

// Transcribe is the transcribe-task token ID.
func (t *Tokenizer) Transcribe() int { return t.transcribe }

// SOTLM is the start-of-language-model token ID.
func (t *Tokenizer) SOTLM() int { return t.sotLM }

// SOTPrev is the start-of-previous-context token ID.
func (t *Tokenizer) SOTPrev() int { return t.sotPrev }

// NoSpeech is the no-speech token ID.
func (t *Tokenizer) NoSpeech() int { return t.noSpeech }

// NoTimestamps is the no-timestamps token ID.
func (t *Tokenizer) NoTimestamps() int { return t.noTimestamps }

// TimestampBegin is the ID of the first 20 ms timestamp token.
func (t *Tokenizer) TimestampBegin() int { return t.timestampBegin }

// LanguageToken returns the token ID for a Whisper language code.
func (t *Tokenizer) LanguageToken(code string) (int, bool) {
	for i := 0; i < whisperLanguageCount; i++ {
		if whisperLanguageCodes[i] == code {
			return t.languageTokens[i], true
		}
	}
	return 0, false
}

// EncodeInto appends the BPE encoding of text to dst. dst must have enough
// capacity for the unmerged UTF-8 bytes in text. It performs no heap allocation.
func (t *Tokenizer) EncodeInto(dst []int, text string) ([]int, error) {
	if t == nil {
		return dst, ErrTokenizerNil
	}
	if containsSpecialToken(text, t.specialIDs) {
		return dst, ErrTokenizerSpecialToken
	}
	for pos := 0; pos < len(text); {
		end := nextPieceEnd(text, pos)
		startOut := len(dst)
		for i := pos; i < end; i++ {
			id, ok := t.encoder[text[i:i+1]]
			if !ok {
				return dst, errors.New("whisper: missing byte token in BPE vocabulary")
			}
			if len(dst) == cap(dst) {
				return dst, ErrTokenizerOutputTooSmall
			}
			dst = append(dst, id)
		}
		for len(dst)-startOut > 1 {
			bestAt, bestID := -1, int(^uint(0)>>1)
			for i := startOut; i+1 < len(dst); i++ {
				if merged, ok := t.merges[mergeKey(dst[i], dst[i+1])]; ok && merged < bestID {
					bestAt, bestID = i, merged
				}
			}
			if bestAt < 0 {
				break
			}
			dst[bestAt] = bestID
			copy(dst[bestAt+1:], dst[bestAt+2:])
			dst = dst[:len(dst)-1]
		}
		pos = end
	}
	return dst, nil
}

// DecodeInto appends decoded token bytes to dst. As in the reference Whisper
// tokenizer, timestamp IDs and later IDs are omitted by the plain decode path.
func (t *Tokenizer) DecodeInto(dst []byte, ids []int) ([]byte, error) {
	if t == nil {
		return dst, ErrTokenizerNil
	}
	for _, id := range ids {
		if id < 0 || id >= t.vocabSize {
			return dst, ErrTokenizerTokenRange
		}
		if id >= t.timestampBegin {
			continue
		}
		if len(dst)+len(t.decoder[id]) > cap(dst) {
			return dst, ErrTokenizerOutputTooSmall
		}
		dst = append(dst, t.decoder[id]...)
	}
	return dst, nil
}

func containsSpecialToken(text string, specials map[string]int) bool {
	for pos := 0; pos+1 < len(text); {
		start := strings.Index(text[pos:], "<|")
		if start < 0 {
			return false
		}
		start += pos
		end := strings.Index(text[start+2:], "|>")
		if end < 0 {
			return false
		}
		end += start + 2
		if _, ok := specials[text[start:end+2]]; ok {
			return true
		}
		pos = end + 2
	}
	return false
}

const (
	classSpace = iota
	classLetter
	classNumber
	classOther
)

func runeClass(text string, pos int) (class, size int) {
	r, size := utf8.DecodeRuneInString(text[pos:])
	switch {
	case unicode.IsLetter(r):
		return classLetter, size
	case unicode.IsNumber(r):
		return classNumber, size
	case unicode.IsSpace(r):
		return classSpace, size
	default:
		return classOther, size
	}
}

func nextPieceEnd(text string, start int) int {
	if end := contractionEnd(text, start); end != 0 {
		return end
	}
	pos := start
	if text[pos] == ' ' && pos+1 < len(text) {
		class, _ := runeClass(text, pos+1)
		if class != classSpace {
			pos++
		}
	}
	class, size := runeClass(text, pos)
	if class == classSpace {
		pos += size
		beforeLast := start
		for pos < len(text) {
			c, w := runeClass(text, pos)
			if c != classSpace {
				break
			}
			beforeLast = pos
			pos += w
		}
		// The reference regex's `\s+(?!\S)|\s+` alternative leaves the
		// final whitespace rune for the next match whenever a run before text
		// has at least two runes. An ASCII final space is then absorbed by the
		// optional-space prefix on that next word or punctuation token.
		if pos < len(text) && beforeLast != start {
			return beforeLast
		}
		return pos
	}
	pos += size
	for pos < len(text) {
		c, w := runeClass(text, pos)
		if c != class {
			break
		}
		pos += w
	}
	return pos
}

func contractionEnd(text string, start int) int {
	if start >= len(text) || text[start] != '\'' {
		return 0
	}
	for _, suffix := range [...]string{"s", "t", "re", "ve", "m", "ll", "d"} {
		end := start + 1 + len(suffix)
		if end > len(text) {
			continue
		}
		match := true
		for i := range suffix {
			c := text[start+1+i]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != suffix[i] {
				match = false
				break
			}
		}
		if match {
			return end
		}
	}
	return 0
}
