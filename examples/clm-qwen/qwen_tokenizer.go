// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package clmqwen

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// QwenTokenizer is the allocation-free warmed tokenizer for the exact
// byte-level BPE pipeline used by Qwen3 tokenizer.json files. It deliberately
// rejects other tokenizer families and pre-tokenizer shapes.
type QwenTokenizer struct {
	byteID [256]int32
	merges map[uint64]qwenMerge
	trie   []qwenAddedNode
	starts [256]bool
}

type qwenMerge struct {
	rank  uint32
	token int32
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
	} `json:"added_tokens"`
	Model struct {
		Type         string           `json:"type"`
		IgnoreMerges bool             `json:"ignore_merges"`
		ByteFallback bool             `json:"byte_fallback"`
		Vocab        map[string]int32 `json:"vocab"`
		Merges       [][]string       `json:"merges"`
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

const qwenSplitPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

var qwenNormReleaseByte = [1]byte{0}

var (
	ErrTokenizerWorkspace = errors.New("clmqwen: nil tokenizer workspace")
	ErrTokenBufferSmall   = errors.New("clmqwen: token output buffer is too small")
)

// LoadQwenTokenizer loads tokenizer.json directly, or from a Hugging Face
// model directory. It validates the model and regex instead of silently
// tokenizing an unsupported family with approximate rules.
func LoadQwenTokenizer(path string) (*QwenTokenizer, error) {
	if filepath.Ext(path) != ".json" {
		path = filepath.Join(path, "tokenizer.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("clmqwen: read tokenizer.json: %w", err)
	}
	var spec qwenTokenizerJSON
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, fmt.Errorf("clmqwen: parse tokenizer.json: %w", err)
	}
	if spec.Model.Type != "BPE" || spec.Model.IgnoreMerges || spec.Model.ByteFallback {
		return nil, fmt.Errorf("clmqwen: unsupported tokenizer model type=%q ignore_merges=%t byte_fallback=%t", spec.Model.Type, spec.Model.IgnoreMerges, spec.Model.ByteFallback)
	}
	if spec.Normalizer.Type != "NFC" {
		return nil, fmt.Errorf("clmqwen: unsupported tokenizer normalizer %q, want NFC", spec.Normalizer.Type)
	}
	var regex string
	for _, node := range spec.PreTokenizer.PreTokenizers {
		if node.Type == "Split" {
			regex = node.Pattern.Regex
			break
		}
	}
	if spec.PreTokenizer.Type != "Sequence" || regex != qwenSplitPattern {
		return nil, fmt.Errorf("clmqwen: unsupported Qwen pre-tokenizer pipeline (type=%q regex=%q)", spec.PreTokenizer.Type, regex)
	}
	if len(spec.Model.Vocab) == 0 || len(spec.Model.Merges) == 0 {
		return nil, errors.New("clmqwen: tokenizer has no BPE vocabulary or merges")
	}

	t := &QwenTokenizer{merges: make(map[uint64]qwenMerge, len(spec.Model.Merges))}
	t.trie = append(t.trie, qwenAddedNode{tokenID: -1})
	var encoded [256]string
	buildQwenByteEncoder(&encoded)
	for b, piece := range encoded {
		id, ok := spec.Model.Vocab[piece]
		if !ok {
			return nil, fmt.Errorf("clmqwen: tokenizer is missing base byte token %q for byte %d", piece, b)
		}
		t.byteID[b] = id
	}
	for rank, pair := range spec.Model.Merges {
		if len(pair) != 2 {
			return nil, fmt.Errorf("clmqwen: merge %d has %d elements, want 2", rank, len(pair))
		}
		left, okLeft := spec.Model.Vocab[pair[0]]
		right, okRight := spec.Model.Vocab[pair[1]]
		merged, okMerged := spec.Model.Vocab[pair[0]+pair[1]]
		if !okLeft || !okRight || !okMerged {
			return nil, fmt.Errorf("clmqwen: merge %d references a token missing from vocab", rank)
		}
		key := qwenPair(left, right)
		if _, duplicate := t.merges[key]; duplicate {
			return nil, fmt.Errorf("clmqwen: duplicate merge pair at rank %d", rank)
		}
		t.merges[key] = qwenMerge{rank: uint32(rank), token: merged}
	}
	for _, added := range spec.AddedTokens {
		if added.Content == "" || added.ID < 0 {
			return nil, fmt.Errorf("clmqwen: invalid added token id=%d content=%q", added.ID, added.Content)
		}
		if err := t.addAddedToken(added.Content, added.ID); err != nil {
			return nil, err
		}
	}
	return t, nil
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
func (t *QwenTokenizer) EncodeInto(text string, dst []int, ws *TokenizerWorkspace) ([]int, error) {
	if t == nil {
		return nil, errors.New("clmqwen: nil Qwen tokenizer")
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

func (t *QwenTokenizer) encodeGap(gap string, out []int, ws *TokenizerWorkspace) ([]int, error) {
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
		start, end := qwenNextPiece(text, i)
		var err error
		out, err = t.encodePiece(text[start:end], out, ws)
		if err != nil {
			return nil, err
		}
		i = end
	}
	return out, nil
}

func (t *QwenTokenizer) encodePiece(piece []byte, out []int, ws *TokenizerWorkspace) ([]int, error) {
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
		if merge, ok := t.merges[qwenPair(ws.token[i], ws.token[i+1])]; ok {
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
		merge, ok := t.merges[qwenPair(ws.token[left], ws.token[right])]
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
				if candidate, ok := t.merges[qwenPair(ws.token[before], ws.token[next])]; ok {
					ws.push(uint64(candidate.rank)<<32 | uint64(uint32(before)))
				}
			}
		}
		if next := ws.next[left]; next >= 0 {
			if candidate, ok := t.merges[qwenPair(ws.token[left], ws.token[next])]; ok {
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

func (t *QwenTokenizer) addAddedToken(content string, id int32) error {
	node := uint32(0)
	for i := 0; i < len(content); i++ {
		b := content[i]
		next := t.trie[node].child[b]
		if next == 0 {
			if uint64(len(t.trie)) >= uint64(^uint32(0)) {
				return errors.New("clmqwen: too many added-token trie nodes")
			}
			t.trie = append(t.trie, qwenAddedNode{tokenID: -1})
			next = uint32(len(t.trie))
			t.trie[node].child[b] = next
		}
		node = next - 1
	}
	if t.trie[node].tokenID >= 0 && t.trie[node].tokenID != id {
		return fmt.Errorf("clmqwen: duplicate added-token content %q", content)
	}
	t.trie[node].tokenID = id
	t.starts[content[0]] = true
	return nil
}

func (t *QwenTokenizer) matchAdded(text string, start int) (int32, int) {
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
func qwenNextPiece(s []byte, start int) (int, int) {
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

	// [^\r\n\p{L}\p{N}]?\p{L}+
	j := start
	if !qwenNewline(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) && start+size < len(s) {
		if next, _ := qwenRuneAt(s, start+size); unicode.IsLetter(next) {
			j += size
		}
	}
	if letter, _ := qwenRuneAt(s, j); unicode.IsLetter(letter) {
		k := j
		for k < len(s) {
			current, step := qwenRuneAt(s, k)
			if !unicode.IsLetter(current) {
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
		if qwenPunctuation(pr) {
			k := p
			for k < len(s) {
				current, step := qwenRuneAt(s, k)
				if !qwenPunctuation(current) {
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

func qwenPunctuation(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}
