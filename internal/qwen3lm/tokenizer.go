// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3lm

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/thesyncim/vibejson"
	"golang.org/x/text/unicode/norm"
)

// Tokenizer is the allocation-free warmed tokenizer for the exact
// byte-level BPE pipeline of Qwen3 tokenizers. It deliberately rejects other
// tokenizer families and pre-tokenizer shapes.
type Tokenizer struct {
	byteID [256]int32
	merges mergeTable
	trie   []qwenAddedNode
	starts [256]bool

	// Decoding: token id's bytes are pieces[pieceEnd[id-1]:pieceEnd[id]],
	// with pieceEnd[-1] taken as zero.
	pieces   []byte
	pieceEnd []uint32
	special  []bool

	// marks makes combining marks letters in the pre-tokenizer, as the
	// Qwen3.5 family's expression (qwenMarksSplitPattern) has them.
	marks bool
}

type qwenMerge struct {
	rank  uint32
	token int32
}

// mergeTable maps a (left, right) token pair to its merge with open
// addressing and linear probing over flat arrays: the encoding hot path does
// no map operations. Slot keys are pair+1 so zero marks an empty slot.
type mergeTable struct {
	keys  []uint64
	vals  []qwenMerge
	shift uint
}

func newMergeTable(n int) mergeTable {
	bits := uint(1)
	for 1<<bits < 2*n {
		bits++
	}
	return mergeTable{keys: make([]uint64, 1<<bits), vals: make([]qwenMerge, 1<<bits), shift: 64 - bits}
}

func (m *mergeTable) slot(key uint64) int {
	return int((key * 0x9e3779b97f4a7c15) >> m.shift)
}

// insert adds key and reports false if it was already present.
func (m *mergeTable) insert(key uint64, v qwenMerge) bool {
	mask := len(m.keys) - 1
	for i := m.slot(key); ; i = (i + 1) & mask {
		switch m.keys[i] {
		case 0:
			m.keys[i], m.vals[i] = key+1, v
			return true
		case key + 1:
			return false
		}
	}
}

func (m *mergeTable) get(key uint64) (qwenMerge, bool) {
	mask := len(m.keys) - 1
	for i := m.slot(key); ; i = (i + 1) & mask {
		switch m.keys[i] {
		case 0:
			return qwenMerge{}, false
		case key + 1:
			return m.vals[i], true
		}
	}
}

// A nonzero child is a trie node index plus one. tokenID is -1 for a
// nonterminal node. The dense byte table makes matching added tokens a single
// indexed load per byte and avoids string construction in EncodeInto.
type qwenAddedNode struct {
	child   [256]uint32
	tokenID int32
}

type qwenTokenizerJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Model struct {
		Type         string           `json:"type"`
		IgnoreMerges bool             `json:"ignore_merges"`
		ByteFallback bool             `json:"byte_fallback"`
		Vocab        map[string]int32 `json:"vocab"`
		Merges       []bpeMerge       `json:"merges"`
	} `json:"model"`
	Normalizer struct {
		Type string `json:"type"`
	} `json:"normalizer"`
	PreTokenizer struct {
		Type          string `json:"type"`
		PreTokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
}

// bpeMerge is one merge of tokenizer.json: a pair of tokens, written
// ["left", "right"] or, as older files have it, "left right".
type bpeMerge [2]string

func (m *bpeMerge) UnmarshalVibeJSON(c vibejson.DecodeCursor) (vibejson.DecodeCursor, error) {
	var pair string
	if s := c; s.String(&pair) == nil {
		left, right, ok := strings.Cut(pair, " ")
		if !ok || left == "" || right == "" || strings.Contains(right, " ") {
			return s, fmt.Errorf("qwen3: malformed merge %q", pair)
		}
		*m = bpeMerge{left, right}
		return s, nil
	}
	if err := c.BeginArray("merge"); err != nil {
		return c, err
	}
	n := 0
	for first := true; ; first = false {
		more, err := c.NextElement(first)
		if err != nil || !more {
			if err == nil && n != 2 {
				err = fmt.Errorf("qwen3: a merge has %d tokens, want 2", n)
			}
			return c, err
		}
		if n == 2 {
			return c, errors.New("qwen3: a merge has more than 2 tokens")
		}
		if err := c.String(&m[n]); err != nil {
			return c, err
		}
		n++
	}
}

// qwenTokenizerConfig is the part of tokenizer_config.json that a slow
// Qwen2Tokenizer (vocab.json plus merges.txt) takes its added tokens from.
type qwenTokenizerConfig struct {
	TokenizerClass string `json:"tokenizer_class"`
	Added          map[string]struct {
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens_decoder"`
}

type qwenAddedToken struct {
	id      int32
	content string
	special bool
}

const qwenSplitPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// qwenMarksSplitPattern is the Qwen3.5 family's expression: combining marks
// (\p{M}) join letters instead of punctuation.
const qwenMarksSplitPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

var qwenNormReleaseByte = [1]byte{0}

var (
	ErrTokenizerWorkspace = errors.New("qwen3: nil tokenizer workspace")
	ErrTokenBufferSmall   = errors.New("qwen3: token output buffer is too small")
)

// LoadTokenizer loads tokenizer.json directly, or a Hugging Face model
// directory holding either tokenizer.json or the vocab.json, merges.txt, and
// tokenizer_config.json of a slow Qwen2Tokenizer (as Qwen3-ASR ships). Both
// describe the same pipeline: NFC, Qwen's split expression, and byte-level
// BPE. It validates the model and regex instead of silently tokenizing an
// unsupported family with approximate rules.
func LoadTokenizer(path string) (*Tokenizer, error) {
	if filepath.Ext(path) != ".json" {
		if _, err := os.Stat(filepath.Join(path, "tokenizer.json")); err != nil {
			return loadQwen2Tokenizer(path)
		}
		path = filepath.Join(path, "tokenizer.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("qwen3: read tokenizer.json: %w", err)
	}
	var spec qwenTokenizerJSON
	if err := vibejson.Unmarshal(raw, &spec); err != nil {
		return nil, fmt.Errorf("qwen3: parse tokenizer.json: %w", err)
	}
	if spec.Model.Type != "BPE" || spec.Model.IgnoreMerges || spec.Model.ByteFallback {
		return nil, fmt.Errorf("qwen3: unsupported tokenizer model type=%q ignore_merges=%t byte_fallback=%t", spec.Model.Type, spec.Model.IgnoreMerges, spec.Model.ByteFallback)
	}
	if spec.Normalizer.Type != "NFC" {
		return nil, fmt.Errorf("qwen3: unsupported tokenizer normalizer %q, want NFC", spec.Normalizer.Type)
	}
	var regex string
	for _, node := range spec.PreTokenizer.PreTokenizers {
		if node.Type == "Split" {
			regex = node.Pattern.Regex
			break
		}
	}
	if spec.PreTokenizer.Type != "Sequence" || regex != qwenSplitPattern && regex != qwenMarksSplitPattern {
		return nil, fmt.Errorf("qwen3: unsupported Qwen pre-tokenizer pipeline (type=%q regex=%q)", spec.PreTokenizer.Type, regex)
	}
	merges := make([][2]string, len(spec.Model.Merges))
	for rank, pair := range spec.Model.Merges {
		merges[rank] = pair
	}
	added := make([]qwenAddedToken, len(spec.AddedTokens))
	for i, a := range spec.AddedTokens {
		added[i] = qwenAddedToken{a.ID, a.Content, a.Special}
	}
	t, err := newTokenizer(spec.Model.Vocab, merges, added)
	if err != nil {
		return nil, err
	}
	t.marks = regex == qwenMarksSplitPattern
	return t, nil
}

// loadQwen2Tokenizer reads a slow Qwen2Tokenizer's files from dir.
func loadQwen2Tokenizer(dir string) (*Tokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return nil, fmt.Errorf("qwen3: no tokenizer.json, and %w", err)
	}
	var config qwenTokenizerConfig
	if err := vibejson.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("qwen3: parse tokenizer_config.json: %w", err)
	}
	if config.TokenizerClass != "Qwen2Tokenizer" && config.TokenizerClass != "Qwen2TokenizerFast" {
		return nil, fmt.Errorf("qwen3: unsupported tokenizer class %q", config.TokenizerClass)
	}
	if raw, err = os.ReadFile(filepath.Join(dir, "vocab.json")); err != nil {
		return nil, fmt.Errorf("qwen3: read vocab.json: %w", err)
	}
	var vocab map[string]int32
	if err := vibejson.Unmarshal(raw, &vocab); err != nil {
		return nil, fmt.Errorf("qwen3: parse vocab.json: %w", err)
	}
	if raw, err = os.ReadFile(filepath.Join(dir, "merges.txt")); err != nil {
		return nil, fmt.Errorf("qwen3: read merges.txt: %w", err)
	}
	text := string(raw)
	var merges [][2]string
	for line := range strings.Lines(text) {
		line = strings.TrimRight(line, "\r\n")
		if line == "" || len(merges) == 0 && strings.HasPrefix(line, "#version:") {
			continue
		}
		left, right, ok := strings.Cut(line, " ")
		if !ok || left == "" || right == "" || strings.Contains(right, " ") {
			return nil, fmt.Errorf("qwen3: malformed merge %d %q", len(merges), line)
		}
		merges = append(merges, [2]string{left, right})
	}
	added := make([]qwenAddedToken, 0, len(config.Added))
	for key, a := range config.Added {
		id, err := strconv.ParseInt(key, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("qwen3: added token id %q: %w", key, err)
		}
		added = append(added, qwenAddedToken{int32(id), a.Content, a.Special})
	}
	return newTokenizer(vocab, merges, added)
}

func newTokenizer(vocab map[string]int32, merges [][2]string, added []qwenAddedToken) (*Tokenizer, error) {
	if len(vocab) == 0 || len(merges) == 0 {
		return nil, errors.New("qwen3: tokenizer has no BPE vocabulary or merges")
	}
	t := &Tokenizer{merges: newMergeTable(len(merges))}
	t.trie = append(t.trie, qwenAddedNode{tokenID: -1})
	var encoded [256]string
	buildQwenByteEncoder(&encoded)
	for b, piece := range encoded {
		id, ok := vocab[piece]
		if !ok {
			return nil, fmt.Errorf("qwen3: tokenizer is missing base byte token %q for byte %d", piece, b)
		}
		t.byteID[b] = id
	}
	for rank, pair := range merges {
		left, okLeft := vocab[pair[0]]
		right, okRight := vocab[pair[1]]
		merged, okMerged := vocab[pair[0]+pair[1]]
		if !okLeft || !okRight || !okMerged {
			return nil, fmt.Errorf("qwen3: merge %d references a token missing from vocab", rank)
		}
		if !t.merges.insert(qwenPair(left, right), qwenMerge{rank: uint32(rank), token: merged}) {
			return nil, fmt.Errorf("qwen3: duplicate merge pair at rank %d", rank)
		}
	}
	for _, a := range added {
		if a.content == "" || a.id < 0 {
			return nil, fmt.Errorf("qwen3: invalid added token id=%d content=%q", a.id, a.content)
		}
		if err := t.addAddedToken(a.content, a.id); err != nil {
			return nil, err
		}
	}
	return t, t.buildDecoder(vocab, added)
}

// buildDecoder lays every token's bytes out in one arena. Vocabulary pieces
// are byte-level strings, mapped back through the inverse byte alphabet;
// added tokens decode to their content.
func (t *Tokenizer) buildDecoder(vocab map[string]int32, added []qwenAddedToken) error {
	var encoded [256]string
	buildQwenByteEncoder(&encoded)
	var alphabet [512]int16
	for i := range alphabet {
		alphabet[i] = -1
	}
	for b, s := range encoded {
		r, _ := utf8.DecodeRuneInString(s)
		alphabet[r] = int16(b)
	}
	n := 0
	for _, id := range vocab {
		n = max(n, int(id)+1)
	}
	for _, a := range added {
		n = max(n, int(a.id)+1)
	}
	if n > 1<<24 {
		return errors.New("qwen3: implausible tokenizer vocabulary size")
	}
	contents := make([]string, n)
	for piece, id := range vocab {
		if id < 0 {
			return fmt.Errorf("qwen3: negative vocabulary id for %q", piece)
		}
		contents[id] = piece
	}
	t.special = make([]bool, n)
	for _, a := range added {
		contents[a.id] = a.content
		t.special[a.id] = a.special
	}
	isAdded := make([]bool, n)
	for _, a := range added {
		isAdded[a.id] = true
	}
	t.pieceEnd = make([]uint32, n)
	for id, s := range contents {
		if isAdded[id] {
			t.pieces = append(t.pieces, s...)
		} else {
			for _, r := range s {
				if r >= rune(len(alphabet)) || alphabet[r] < 0 {
					return fmt.Errorf("qwen3: vocabulary piece %q of token %d is outside the byte alphabet", s, id)
				}
				t.pieces = append(t.pieces, byte(alphabet[r]))
			}
		}
		t.pieceEnd[id] = uint32(len(t.pieces))
	}
	return nil
}

// Tokens reports the number of token ids the tokenizer can decode.
func (t *Tokenizer) Tokens() int { return len(t.pieceEnd) }

// AddedID returns the id of the added token whose content is exactly text,
// such as "<|im_end|>".
func (t *Tokenizer) AddedID(text string) (int, bool) {
	id, end := t.matchAdded(text, 0)
	if id < 0 || end != len(text) {
		return 0, false
	}
	return int(id), true
}

// Special reports whether id is a special added token, which decoding with
// skipSpecial drops.
func (t *Tokenizer) Special(id int) bool { return id >= 0 && id < len(t.special) && t.special[id] }

// Piece returns the bytes of one token: a vocabulary token's raw bytes,
// which may be part of a UTF-8 sequence, or an added token's content. Ids the
// tokenizer does not know yield nil. The slice aliases the tokenizer.
func (t *Tokenizer) Piece(id int) []byte {
	if id < 0 || id >= len(t.pieceEnd) {
		return nil
	}
	start := uint32(0)
	if id > 0 {
		start = t.pieceEnd[id-1]
	}
	return t.pieces[start:t.pieceEnd[id]:t.pieceEnd[id]]
}

// DecodeAppend appends the text of ids to dst, as Hugging Face's decode does
// with errors="replace" and clean_up_tokenization_spaces=False: token bytes
// are concatenated, and each maximal invalid UTF-8 subpart becomes U+FFFD.
// With skipSpecial, special added tokens are dropped. Ids the tokenizer does
// not know decode to nothing. It allocates only when dst lacks capacity.
func (t *Tokenizer) DecodeAppend(dst []byte, ids []int, skipSpecial bool) []byte {
	start := len(dst)
	for _, id := range ids {
		if skipSpecial && t.Special(id) {
			continue
		}
		dst = append(dst, t.Piece(id)...)
	}
	if utf8.Valid(dst[start:]) {
		return dst
	}
	// Write the repaired text after the raw bytes, then move it into place.
	end := len(dst)
	dst = appendReplacingInvalid(dst, dst[start:end])
	n := copy(dst[start:], dst[end:])
	return dst[:start+n]
}

// appendReplacingInvalid appends src to dst with each maximal subpart of an
// ill-formed UTF-8 sequence replaced by U+FFFD, as Python's
// bytes.decode("utf-8", errors="replace") does.
func appendReplacingInvalid(dst, src []byte) []byte {
	for i := 0; i < len(src); {
		if src[i] < utf8.RuneSelf {
			dst = append(dst, src[i])
			i++
			continue
		}
		if r, size := utf8.DecodeRune(src[i:]); r != utf8.RuneError || size > 1 {
			dst = append(dst, src[i:i+size]...)
			i += size
			continue
		}
		i += invalidUTF8Span(src[i:])
		dst = append(dst, "\uFFFD"...)
	}
	return dst
}

// invalidUTF8Span returns the length of the maximal subpart of the ill-formed
// sequence at the start of b: a valid lead byte and the valid continuation
// bytes that follow it, or one byte.
func invalidUTF8Span(b []byte) int {
	need, lo, hi := 0, byte(0x80), byte(0xBF)
	switch lead := b[0]; {
	case lead >= 0xC2 && lead <= 0xDF:
		need = 1
	case lead == 0xE0:
		need, lo = 2, 0xA0
	case lead == 0xED:
		need, hi = 2, 0x9F
	case lead >= 0xE1 && lead <= 0xEF:
		need = 2
	case lead == 0xF0:
		need, lo = 3, 0x90
	case lead == 0xF4:
		need, hi = 3, 0x8F
	case lead >= 0xF1 && lead <= 0xF3:
		need = 3
	default:
		return 1
	}
	n := 1
	for ; n <= need && n < len(b); n++ {
		if c := b[n]; c < lo || c > hi {
			break
		}
		lo, hi = 0x80, 0xBF
	}
	return n
}

// TokenizerWorkspace owns reusable normalization, BPE-node, and heap storage.
// A workspace must not be used concurrently. Its capacity grows to fit the
// longest text seen so far; repeating that workload performs no allocations.
type TokenizerWorkspace struct {
	normalized []byte
	normalizer norm.Iter
	token      []int32
	prev       []int32
	next       []int32
	heap       []uint64
}

// EncodeInto writes token IDs into dst. dst must have enough capacity for the
// complete token sequence; no fallback allocation is made when it is too
// small. It adds no BOS token, matching the CLM Qwen3 reference path.
func (t *Tokenizer) EncodeInto(text string, dst []int, ws *TokenizerWorkspace) ([]int, error) {
	if t == nil {
		return nil, errors.New("qwen3: nil Qwen tokenizer")
	}
	if ws == nil {
		return nil, ErrTokenizerWorkspace
	}
	out := dst[:0]
	gapStart := 0
	for i := 0; i < len(text); {
		if t.starts[text[i]] {
			if id, end := t.matchAdded(text, i); end > i {
				var err error
				out, err = t.encodeGap(text[gapStart:i], out, ws)
				if err != nil {
					return nil, err
				}
				if len(out) == cap(out) {
					return nil, ErrTokenBufferSmall
				}
				out = append(out, int(id))
				gapStart = end
				i = end
				continue
			}
		}
		_, size := utf8.DecodeRuneInString(text[i:])
		if size < 1 {
			size = 1
		}
		i += size
	}
	var err error
	out, err = t.encodeGap(text[gapStart:], out, ws)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (t *Tokenizer) encodeGap(gap string, out []int, ws *TokenizerWorkspace) ([]int, error) {
	if gap == "" {
		return out, nil
	}
	if norm.NFC.QuickSpanString(gap) == len(gap) {
		ws.normalized = append(ws.normalized[:0], gap...)
	} else {
		ws.normalizer.InitString(norm.NFC, gap)
		ws.normalized = ws.normalized[:0]
		for !ws.normalizer.Done() {
			ws.normalized = append(ws.normalized, ws.normalizer.Next()...)
		}
		// Iter retains its input string. Rebind it to a static byte so a
		// long caller string is not kept alive by the reusable workspace.
		ws.normalizer.Init(norm.NFC, qwenNormReleaseByte[:])
	}
	text := ws.normalized
	for i := 0; i < len(text); {
		start, end := qwenNextPiece(text, i, t.marks)
		var err error
		out, err = t.encodePiece(text[start:end], out, ws)
		if err != nil {
			return nil, err
		}
		i = end
	}
	return out, nil
}

func (t *Tokenizer) encodePiece(piece []byte, out []int, ws *TokenizerWorkspace) ([]int, error) {
	n := len(piece)
	if n == 0 {
		return out, nil
	}
	ws.ensureBPE(n)
	for i, b := range piece {
		ws.token[i] = t.byteID[b]
		ws.prev[i] = int32(i - 1)
		if i+1 == n {
			ws.next[i] = -1
		} else {
			ws.next[i] = int32(i + 1)
		}
	}
	ws.heap = ws.heap[:0]
	for i := 0; i+1 < n; i++ {
		if merge, ok := t.merges.get(qwenPair(ws.token[i], ws.token[i+1])); ok {
			ws.push(uint64(merge.rank)<<32 | uint64(uint32(i)))
		}
	}
	for len(ws.heap) != 0 {
		key := ws.pop()
		left := int(uint32(key))
		if left >= n {
			continue
		}
		right := int(ws.next[left])
		if right < 0 {
			continue
		}
		merge, ok := t.merges.get(qwenPair(ws.token[left], ws.token[right]))
		if !ok || uint64(merge.rank)<<32|uint64(uint32(left)) != key {
			continue
		}
		ws.token[left] = merge.token
		oldRight := right
		ws.next[left] = ws.next[oldRight]
		if after := ws.next[oldRight]; after >= 0 {
			ws.prev[after] = int32(left)
		}
		ws.next[oldRight] = -1
		ws.prev[oldRight] = -1
		if before := ws.prev[left]; before >= 0 {
			if next := ws.next[before]; next >= 0 {
				if candidate, ok := t.merges.get(qwenPair(ws.token[before], ws.token[next])); ok {
					ws.push(uint64(candidate.rank)<<32 | uint64(uint32(before)))
				}
			}
		}
		if next := ws.next[left]; next >= 0 {
			if candidate, ok := t.merges.get(qwenPair(ws.token[left], ws.token[next])); ok {
				ws.push(uint64(candidate.rank)<<32 | uint64(uint32(left)))
			}
		}
	}
	for i := int32(0); i >= 0; i = ws.next[i] {
		if len(out) == cap(out) {
			return nil, ErrTokenBufferSmall
		}
		out = append(out, int(ws.token[i]))
	}
	return out, nil
}

func (ws *TokenizerWorkspace) ensureBPE(n int) {
	if cap(ws.token) < n {
		ws.token = make([]int32, n)
		ws.prev = make([]int32, n)
		ws.next = make([]int32, n)
	} else {
		ws.token = ws.token[:n]
		ws.prev = ws.prev[:n]
		ws.next = ws.next[:n]
	}
	if cap(ws.heap) < 3*n {
		ws.heap = make([]uint64, 0, 3*n)
	}
}

func (ws *TokenizerWorkspace) push(key uint64) {
	h := ws.heap
	h = append(h, key)
	i := len(h) - 1
	for i > 0 {
		p := (i - 1) >> 1
		if h[p] <= key {
			break
		}
		h[i] = h[p]
		i = p
	}
	h[i] = key
	ws.heap = h
}

func (ws *TokenizerWorkspace) pop() uint64 {
	h := ws.heap
	root := h[0]
	last := h[len(h)-1]
	h = h[:len(h)-1]
	if len(h) != 0 {
		i := 0
		for {
			left := 2*i + 1
			if left >= len(h) {
				break
			}
			smallest := left
			if right := left + 1; right < len(h) && h[right] < h[left] {
				smallest = right
			}
			if h[smallest] >= last {
				break
			}
			h[i] = h[smallest]
			i = smallest
		}
		h[i] = last
	}
	ws.heap = h
	return root
}

func (t *Tokenizer) addAddedToken(content string, id int32) error {
	node := uint32(0)
	for i := 0; i < len(content); i++ {
		b := content[i]
		next := t.trie[node].child[b]
		if next == 0 {
			if uint64(len(t.trie)) >= uint64(^uint32(0)) {
				return errors.New("qwen3: too many added-token trie nodes")
			}
			t.trie = append(t.trie, qwenAddedNode{tokenID: -1})
			next = uint32(len(t.trie))
			t.trie[node].child[b] = next
		}
		node = next - 1
	}
	if t.trie[node].tokenID >= 0 && t.trie[node].tokenID != id {
		return fmt.Errorf("qwen3: duplicate added-token content %q", content)
	}
	t.trie[node].tokenID = id
	t.starts[content[0]] = true
	return nil
}

func (t *Tokenizer) matchAdded(text string, start int) (int32, int) {
	node := uint32(0)
	var id int32 = -1
	end := start
	for i := start; i < len(text); i++ {
		next := t.trie[node].child[text[i]]
		if next == 0 {
			break
		}
		node = next - 1
		if candidate := t.trie[node].tokenID; candidate >= 0 {
			id = candidate
			end = i + 1
		}
	}
	return id, end
}

func qwenPair(left, right int32) uint64 {
	return uint64(uint32(left))<<32 | uint64(uint32(right))
}

func buildQwenByteEncoder(encoded *[256]string) {
	n := 0
	for b := range 256 {
		r := rune(b)
		if !((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
			r = rune(256 + n)
			n++
		}
		encoded[b] = string(r)
	}
}

// qwenNextPiece returns the next ordered-alternative match from Qwen's exact
// pre-tokenizer expression. The implementation walks UTF-8 in place and does
// not build rune slices or substrings.
func qwenNextPiece(s []byte, start int, marks bool) (int, int) {
	r, size := qwenRuneAt(s, start)
	if r == '\'' && start+size < len(s) {
		r1, sz1 := qwenRuneAt(s, start+size)
		switch unicode.ToLower(r1) {
		case 's', 't', 'm', 'd':
			return start, start + size + sz1
		case 'r', 'v', 'l':
			nextPos := start + size + sz1
			if nextPos < len(s) {
				r2, sz2 := qwenRuneAt(s, nextPos)
				c1, c2 := unicode.ToLower(r1), unicode.ToLower(r2)
				if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
					return start, nextPos + sz2
				}
			}
		}
	}

	// [^\r\n\p{L}\p{N}]?\p{L}+, or [\p{L}\p{M}]+ with marks
	j := start
	if !qwenNewline(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) && start+size < len(s) {
		if next, _ := qwenRuneAt(s, start+size); qwenLetter(next, marks) {
			j += size
		}
	}
	if letter, _ := qwenRuneAt(s, j); qwenLetter(letter, marks) {
		k := j
		for k < len(s) {
			current, step := qwenRuneAt(s, k)
			if !qwenLetter(current, marks) {
				break
			}
			k += step
		}
		return start, k
	}

	// \p{N} is a single rune in Qwen3's expression.
	if unicode.IsNumber(r) {
		return start, start + size
	}

	//  ?[^\s\p{L}\p{N}]+[\r\n]*
	p := start
	if r == ' ' {
		p += size
	}
	if p < len(s) {
		pr, _ := qwenRuneAt(s, p)
		if qwenPunctuation(pr, marks) {
			k := p
			for k < len(s) {
				current, step := qwenRuneAt(s, k)
				if !qwenPunctuation(current, marks) {
					break
				}
				k += step
			}
			for k < len(s) {
				current, step := qwenRuneAt(s, k)
				if !qwenNewline(current) {
					break
				}
				k += step
			}
			return start, k
		}
	}

	if unicode.IsSpace(r) {
		w := start
		lastNewline := -1
		for w < len(s) {
			current, step := qwenRuneAt(s, w)
			if !unicode.IsSpace(current) {
				break
			}
			if qwenNewline(current) {
				lastNewline = w
			}
			w += step
		}
		if lastNewline >= 0 {
			_, step := qwenRuneAt(s, lastNewline)
			return start, lastNewline + step
		}
		if w == len(s) {
			return start, w
		}
		if w-start > size {
			// Keep the final space for the next letter or punctuation alternative.
			last := w
			_, step := utf8.DecodeLastRune(s[start:w])
			return start, last - step
		}
		return start, w
	}

	// The alternatives cover every valid Unicode rune. Keep invalid UTF-8
	// progressing one byte at a time, just as the reference tokenizer does.
	return start, start + size
}

func qwenRuneAt(s []byte, i int) (rune, int) {
	r, size := utf8.DecodeRune(s[i:])
	if size == 0 {
		return utf8.RuneError, 1
	}
	return r, size
}

func qwenNewline(r rune) bool { return r == '\r' || r == '\n' }

func qwenPunctuation(r rune, marks bool) bool {
	return !unicode.IsSpace(r) && !qwenLetter(r, marks) && !unicode.IsNumber(r)
}

// qwenLetter reports whether r is \p{L}, or with marks [\p{L}\p{M}].
func qwenLetter(r rune, marks bool) bool {
	return unicode.IsLetter(r) || marks && unicode.Is(unicode.M, r)
}
