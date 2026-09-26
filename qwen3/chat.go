// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/internal/safetensors"
	"github.com/thesyncim/vibejson"
)

// Chat generates text with a Qwen3 model: it is a chat.Generator whose
// sessions keep their conversation's keys and values evaluated, so each new
// message costs only its own tokens. Replies use Qwen3's non-thinking mode.
// Sessions share the model; their calls are serialized.
type Chat struct {
	mu      sync.Mutex
	weights *Weights
	tokens  *Tokenizer
	eval    *Evaluator
	ws      *Workspace
	closed  bool
	own     bool   // Close releases the weights
	path    string // the snapshot, when OpenChat loaded it

	imEnd, endText int
	// enders are the end marker and the tokens of only closing
	// punctuation (closer) or white space: text ends with any of them.
	enders  []int
	closer  []bool
	newline []int
	header  [4][]int // "<|im_start|>system\n" and so on, by chat.Role
	answer  []int    // the assistant header and the empty thinking block
	// Tool calls: the tokens around a call, and after a tool's result.
	callOpen, callClose int
	resultEnd           []int
}

var _ chat.Generator = (*Chat)(nil)

// OpenChat loads the Qwen3 snapshot at path with its language-model head,
// for text generation. Options select the weight format and threads; the
// caches of Open do not apply.
func OpenChat(path string, opts Options) (*Chat, error) {
	if opts.Threads < 0 {
		return nil, fmt.Errorf("qwen3: invalid thread count %d", opts.Threads)
	}
	tokens, err := LoadTokenizer(path)
	if err != nil {
		return nil, fmt.Errorf("qwen3: load tokenizer: %w", err)
	}
	st, err := safetensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("qwen3: %w", err)
	}
	head := "lm_head.weight"
	if !st.Has(head) {
		head = "model.embed_tokens.weight" // smaller Qwen3 models tie it
	}
	st.Close()
	weights, err := LoadWeightsOptions(path, LoadOptions{Format: opts.Weights, Head: head})
	if err != nil {
		return nil, err
	}
	c, err := NewChat(weights, tokens, opts.threads())
	if err != nil {
		weights.Release()
		return nil, err
	}
	c.own, c.path = true, path
	if !thinks(path) {
		c.answer = slices.Clone(c.header[chat.Assistant])
	}
	return c, nil
}

// thinks reports whether the chat template of the snapshot at path has
// Qwen3's thinking block, which a non-thinking reply opens empty. The
// Instruct-2507 models have none: their replies start after the header.
func thinks(path string) bool {
	raw, err := os.ReadFile(filepath.Join(path, "tokenizer_config.json"))
	if err != nil {
		return true
	}
	var cfg struct {
		Template string `json:"chat_template"`
	}
	if vibejson.Unmarshal(raw, &cfg) != nil || cfg.Template == "" {
		return true
	}
	return strings.Contains(cfg.Template, "<think>")
}

// Questions returns a Model for embeddings and zero-shot questions that
// shares c's weights, so one loaded model serves both. It needs a Chat
// from OpenChat. Close it before c; closing it leaves the weights loaded.
func (c *Chat) Questions(opts Options) (*Model, error) {
	if c.path == "" {
		return nil, errors.New("qwen3: Questions needs a Chat from OpenChat")
	}
	cfg := c.weights.Config()
	letters, err := loadLetterHead(c.path, c.tokens, cfg.Hidden, cfg.Vocab)
	if err != nil {
		return nil, err
	}
	entries := opts.CacheEntries
	if entries == 0 {
		entries = defaultCacheEntries
	}
	prefix := opts.PrefixCacheTokens
	if prefix == 0 {
		prefix = maxTokens
	}
	m, err := newModel(c.weights, c.tokens, opts.threads(), entries, min(prefix, maxTokens, cfg.MaxPositions))
	if err != nil {
		return nil, err
	}
	m.letters = letters
	if !thinks(c.path) {
		m.answer = answerPlain
	}
	return m, nil
}

// NewChat generates with weights loaded with their head and the tokenizer
// of the same snapshot, using threads CPU workers. Close leaves the weights
// loaded.
func NewChat(weights *Weights, tokens *Tokenizer, threads int) (*Chat, error) {
	eval, err := NewEvaluator(weights)
	if err != nil {
		return nil, err
	}
	ws, err := eval.NewWorkspace(threads)
	if err != nil {
		return nil, err
	}
	c := &Chat{weights: weights, tokens: tokens, eval: eval, ws: ws}
	var ok1, ok2 bool
	c.imEnd, ok1 = tokens.AddedID("<|im_end|>")
	c.endText, ok2 = tokens.AddedID("<|endoftext|>")
	if !ok1 || !ok2 {
		return nil, errors.New("qwen3: the tokenizer lacks <|im_end|> or <|endoftext|>")
	}
	var tws TokenizerWorkspace
	encode := func(s string) []int {
		if err != nil {
			return nil
		}
		var ids []int
		ids, err = tokens.EncodeInto(s, make([]int, 0, 2*len(s)+8), &tws)
		return ids
	}
	c.newline = encode("\n")
	c.header[chat.System] = encode("<|im_start|>system\n")
	c.header[chat.User] = encode("<|im_start|>user\n")
	c.header[chat.Assistant] = encode("<|im_start|>assistant\n")
	c.header[chat.ToolResult] = encode("<|im_start|>user\n<tool_response>\n")
	c.resultEnd = encode("\n</tool_response>")
	c.answer = encode(answerThinking)
	if err != nil {
		return nil, fmt.Errorf("qwen3: chat template: %w", err)
	}
	c.callOpen, ok1 = tokens.AddedID("<tool_call>")
	c.callClose, ok2 = tokens.AddedID("</tool_call>")
	if !ok1 || !ok2 {
		return nil, errors.New("qwen3: the tokenizer lacks <tool_call> or </tool_call>")
	}
	c.enders = append(c.enders, c.imEnd)
	c.closer = make([]bool, weights.Config().Vocab)
	for id := range c.closer {
		if closes, blank := ending(tokens.Piece(id)); closes || blank {
			c.enders = append(c.enders, id)
			c.closer[id] = closes
		}
	}
	if err := c.warm(); err != nil {
		return nil, err
	}
	return c, nil
}

// warm runs a prompt and a decoding step once, so that the GPU's first-use
// costs (scratch allocation, first dispatch of every kernel) are paid while
// loading instead of by the first reply.
func (c *Chat) warm() error {
	cfg := c.weights.Config()
	kv, err := c.eval.NewPrefixKV(64)
	if err != nil {
		return err
	}
	prompt := make([]int, 48)
	for i := range prompt {
		prompt[i] = c.answer[i%len(c.answer)]
	}
	hidden, logits := make([]float32, cfg.Hidden), make([]float32, cfg.Vocab)
	if err := c.eval.HiddenLastExtendInto(kv, 0, prompt, hidden, c.ws); err != nil {
		return err
	}
	if err := c.eval.HiddenLastExtendInto(kv, len(prompt), prompt[:1], hidden, c.ws); err != nil {
		return err
	}
	return c.eval.LogitsInto(hidden, logits, c.ws)
}

// Close releases the workspace, and the weights when OpenChat loaded them.
// Close sessions first.
func (c *Chat) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	err := c.ws.Close()
	if c.own {
		c.weights.Release()
	}
	return err
}

// NewSession starts a conversation.
func (c *Chat) NewSession(system string, tools ...chat.ToolSpec) (chat.Session, error) {
	cfg := c.weights.Config()
	s := &Session{c: c, hidden: make([]float32, cfg.Hidden), logits: make([]float32, cfg.Vocab), reply: -1}
	if len(tools) > 0 {
		var err error
		if system, err = toolsPrompt(system, tools); err != nil {
			return nil, err
		}
		// What follows <tool_call> up to each tool's arguments, drafted
		// whole: "\n{"name": "stay_quiet", "arguments":".
		for _, t := range tools {
			text := "\n{\"name\": \"" + t.Name + "\", \"arguments\":"
			ids, err := c.tokens.EncodeInto(text, make([]int, 0, len(text)), &s.tws)
			if err != nil {
				return nil, err
			}
			s.tools = append(s.tools, toolDraft{name: t.Name, scaffold: ids})
		}
	}
	if system != "" {
		if err := s.Add(chat.System, system); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// Session is one conversation of a Chat; see chat.Session.
type Session struct {
	c     *Chat
	kv    *PrefixKV
	ids   []int // the conversation's tokens; kv holds a prefix of them
	probe []int // Finished's tokens
	// ready is the number of stored tokens after which hidden is the model
	// state, so that a Reply they begin samples at once.
	ready    int
	finished float32   // Finished's probability for the tokens ready marks
	tail     []float32 // Finished's states
	reply    int       // where the last reply's tokens start in ids, or -1
	hidden   []float32
	logits   []float32
	tws      TokenizerWorkspace
	text     []byte // message rendering and decoded pieces
	sample   sampler
	closed   bool

	// Tool calls: the tools offered, the calls of the last reply with
	// their arguments' storage, and the call being written.
	tools   []toolDraft
	calls   []chat.Call
	args    [][]byte
	calling bool
	call    []byte    // its text
	callIDs []int     // its tokens
	vlogits []float32 // the logits of drafted states

	// The tokens of the reply so far, for the presence penalty.
	seen    []bool
	seenIDs []int
}

// toolDraft is a tool's name and the tokens a call to it starts with.
type toolDraft struct {
	name     string
	scaffold []int
}

var _ chat.Session = (*Session)(nil)

// Add appends a complete message.
func (s *Session) Add(role chat.Role, text string) error {
	if s.closed {
		return chat.ErrClosed
	}
	if role > chat.ToolResult {
		return fmt.Errorf("qwen3: unknown role %d", role)
	}
	c := s.c
	s.ids = append(s.ids, c.header[role]...)
	start := len(s.ids)
	// A token takes at least one byte; the end marker and newline follow.
	if need := start + len(text) + len(c.resultEnd) + 1 + len(c.newline); cap(s.ids) < need {
		s.ids = append(make([]int, 0, 2*need), s.ids...)
	}
	body, err := c.tokens.EncodeInto(text, s.ids[start:start], &s.tws)
	if err != nil {
		s.ids = s.ids[:start-len(c.header[role])]
		return fmt.Errorf("qwen3: tokenize message: %w", err)
	}
	s.ids = s.ids[:start+len(body)]
	if role == chat.ToolResult {
		s.ids = append(s.ids, c.resultEnd...)
	}
	s.ids = append(s.ids, c.imEnd)
	s.ids = append(s.ids, c.newline...)
	s.reply = -1
	return nil
}

// Calls returns the tool calls of the last reply; see chat.Session.
func (s *Session) Calls() []chat.Call { return s.calls }

// Reply generates the assistant's next message.
func (s *Session) Reply(ctx context.Context, opts chat.Options, w io.Writer) error {
	if s.closed {
		return chat.ErrClosed
	}
	c := s.c
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return chat.ErrClosed
	}
	s.ids = append(s.ids, c.answer...)
	s.reply = len(s.ids)
	limit := c.weights.Config().MaxPositions
	room := limit - len(s.ids) - 2
	if opts.MaxTokens > 0 {
		room = min(room, opts.MaxTokens)
	}
	if room <= 0 {
		return errors.New("qwen3: the conversation fills the context")
	}
	// Room for the prompt and a typical reply; longer replies grow it.
	if err := s.reserve(len(s.ids)+min(room, 256)+2, limit); err != nil {
		return err
	}
	if s.ready != len(s.ids) || len(s.kv.Tokens()) != len(s.ids) || s.kv.CommonPrefix(s.ids) != len(s.ids) {
		keep := min(s.kv.CommonPrefix(s.ids), len(s.ids)-1)
		if err := c.eval.HiddenLastExtendInto(s.kv, keep, s.ids[keep:], s.hidden, c.ws); err != nil {
			s.ready = 0
			return err
		}
	}
	s.ready = 0
	s.sample.reset(opts)
	s.text = s.text[:0]
	s.calls, s.calling = s.calls[:0], false
	s.forget()
	var err error
	next, sampled := 0, false // sampled: next was drawn while checking a draft
	for n := 0; n < room; n++ {
		if !sampled {
			if err = c.eval.LogitsInto(s.hidden, s.logits, c.ws); err != nil {
				break
			}
			s.penalize(s.logits, opts)
			next = s.sample.next(s.logits, opts)
		}
		sampled = false
		if next == c.imEnd || next == c.endText {
			break
		}
		if err = s.take(next, w); err != nil || n+1 == room {
			break
		}
		if err = ctx.Err(); err != nil {
			break
		}
		draft := s.draft(room - n - 1)
		if need := len(s.ids) + len(draft) + 2; need > s.kv.Capacity() {
			if err = s.reserve(max(2*s.kv.Capacity(), need), limit); err != nil {
				break
			}
		}
		if len(draft) == 0 {
			if err = c.eval.HiddenLastExtendInto(s.kv, len(s.ids)-1, s.ids[len(s.ids)-1:], s.hidden, c.ws); err != nil {
				break
			}
			continue
		}
		var taken int
		if taken, next, err = s.verify(draft, opts, w); err != nil {
			break
		}
		n += taken
		sampled = true
	}
	if ferr := s.emit(nil, w, true); err == nil {
		err = ferr
	}
	s.ids = append(s.ids, c.imEnd)
	s.ids = append(s.ids, c.newline...)
	// Grow the store now, while the reply is being heard, rather than when
	// the next message must be judged at once.
	if s.kv.Capacity()-len(s.ids) < roomAhead {
		if gerr := s.reserve(len(s.ids)+roomAhead, limit); err == nil {
			err = gerr
		}
	}
	return err
}

// roomAhead is the room a conversation keeps for its next message and the
// reply's start, grown after a reply when it runs short.
const roomAhead = 512

// take appends a token of the reply: text goes to w, and a tool call's
// tokens are collected until it closes, when the call is recorded.
func (s *Session) take(id int, w io.Writer) error {
	c := s.c
	s.ids = append(s.ids, id)
	s.remember(id)
	switch {
	case id == c.callOpen:
		s.calling, s.call, s.callIDs = true, s.call[:0], s.callIDs[:0]
	case id == c.callClose:
		if s.calling {
			s.calling = false
			s.record()
		}
	case s.calling:
		s.call = append(s.call, c.tokens.Piece(id)...)
		s.callIDs = append(s.callIDs, id)
	default:
		return s.emit(c.tokens.Piece(id), w, false)
	}
	return nil
}

// remember records a token of the reply, for the presence penalty.
func (s *Session) remember(id int) {
	if len(s.seen) == 0 {
		s.seen = make([]bool, s.c.weights.Config().Vocab)
	}
	if !s.seen[id] {
		s.seen[id] = true
		s.seenIDs = append(s.seenIDs, id)
	}
}

// forget clears the tokens remembered for the presence penalty.
func (s *Session) forget() {
	for _, id := range s.seenIDs {
		s.seen[id] = false
	}
	s.seenIDs = s.seenIDs[:0]
}

// penalize lowers the logits of the tokens the reply already holds.
func (s *Session) penalize(logits []float32, opts chat.Options) {
	if opts.Presence == 0 {
		return
	}
	for _, id := range s.seenIDs {
		logits[id] -= opts.Presence
	}
}

// record adds the call just written to Calls, if it is well formed.
func (s *Session) record() {
	name, args, err := parseCall(s.call)
	if err != nil {
		return
	}
	i := len(s.calls)
	if i == len(s.args) {
		s.args = append(s.args, nil)
	}
	s.args[i] = append(s.args[i][:0], args...)
	call := chat.Call{Arguments: s.args[i]}
	for _, t := range s.tools {
		if string(name) == t.name {
			call.Name = t.name
		}
	}
	if call.Name == "" {
		call.Name = string(name)
	}
	s.calls = append(s.calls, call)
}

// draft returns, while a tool call is being written, the tokens it is sure
// to go on with, up to max: the rest of the scaffold that every tool it can
// still be calling shares.
func (s *Session) draft(max int) []int {
	if !s.calling || max <= 1 {
		return nil
	}
	var d []int
	some := false
	for _, t := range s.tools {
		n := len(s.callIDs)
		if n >= len(t.scaffold) || !slices.Equal(t.scaffold[:n], s.callIDs) {
			continue
		}
		rest := t.scaffold[n:]
		if !some {
			d, some = rest, true
			continue
		}
		k := 0
		for k < len(d) && k < len(rest) && d[k] == rest[k] {
			k++
		}
		d = d[:k]
	}
	return d[:min(len(d), max-1)]
}

// verify evaluates the last token taken and a draft after it in one pass,
// then samples each position as decoding one token at a time would, taking
// drafted tokens while the samples agree. It returns how many it took and
// the first sample that is not a drafted token, which comes next.
func (s *Session) verify(draft []int, opts chat.Options, w io.Writer) (taken, next int, err error) {
	c, h, vocab := s.c, len(s.hidden), len(s.logits)
	base := len(s.ids) - 1
	s.probe = append(append(s.probe[:0], s.ids[base]), draft...)
	k := len(s.probe)
	s.tail = slices.Grow(s.tail[:0], k*h)[:k*h]
	if err := c.eval.HiddenTailExtendEmbedInto(s.kv, base, s.probe, Embeds{}, s.tail, c.ws); err != nil {
		return 0, 0, err
	}
	s.vlogits = slices.Grow(s.vlogits[:0], k*vocab)[:k*vocab]
	if err := c.eval.LogitsRowsInto(s.tail, s.vlogits, c.ws); err != nil {
		return 0, 0, err
	}
	for i := range k {
		logits := s.vlogits[i*vocab : (i+1)*vocab]
		s.penalize(logits, opts)
		x := s.sample.next(logits, opts)
		if i == len(draft) || x != draft[i] {
			copy(s.hidden, s.tail[i*h:(i+1)*h])
			return taken, x, nil
		}
		if err := s.take(x, w); err != nil {
			return taken, 0, err
		}
		taken++
	}
	return taken, 0, errors.New("qwen3: unreachable")
}

// Finished evaluates the conversation followed by a message of role holding
// text, and returns the probability that the message ends after the text:
// that its words end there, with its end token or closing punctuation, and
// that the message ends after the closing punctuation it has, if any. A
// fragment a recognizer punctuated ("Can you tell me a?") ends no words,
// and a statement a question follows ("I'm going hiking.") ends no message. The same pass evaluates the end and the assistant's header, as a
// reply to the message would begin, so that Add and Reply next start
// sampling at once.
func (s *Session) Finished(ctx context.Context, role chat.Role, text string) (float32, error) {
	if s.closed {
		return 0, chat.ErrClosed
	}
	if role > chat.Assistant {
		return 0, fmt.Errorf("qwen3: unknown role %d", role)
	}
	c := s.c
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, chat.ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	probe := append(append(s.probe[:0], s.ids...), c.header[role]...)
	start := len(probe)
	if need := start + len(text); cap(probe) < need {
		probe = append(make([]int, 0, 2*need), probe...)
	}
	body, err := c.tokens.EncodeInto(text, probe[start:start], &s.tws)
	if err != nil {
		return 0, fmt.Errorf("qwen3: tokenize message: %w", err)
	}
	probe = probe[:start+len(body)]
	end := len(probe) // the state after the text predicts the end
	closed := 0       // closing punctuation the text ends with
	for closed < len(body) && c.closer[body[len(body)-1-closed]] {
		closed++
	}
	probe = append(probe, c.imEnd)
	probe = append(probe, c.newline...)
	probe = append(probe, c.answer...)
	s.probe = probe
	limit := c.weights.Config().MaxPositions
	if len(probe)+4 > limit {
		return 0, errors.New("qwen3: the conversation fills the context")
	}
	// The same message judged again, as a transcript that ends as it was
	// heard while spoken, costs nothing.
	if s.ready == len(probe) && len(s.kv.Tokens()) == len(probe) && s.kv.CommonPrefix(probe) == len(probe) {
		return s.finished, nil
	}
	if err := s.reserve(min(len(probe)+256+4, limit), limit); err != nil {
		return 0, err
	}
	// The states from the last word's on: before the punctuation, after
	// it, and the header's last, which a reply starts from.
	keep := min(s.kv.CommonPrefix(probe), end-1-closed)
	h := len(s.hidden)
	k := len(probe) - end + 1 + closed
	s.tail = slices.Grow(s.tail[:0], k*h)[:k*h]
	if err := c.eval.HiddenTailExtendEmbedInto(s.kv, keep, probe[keep:], Embeds{}, s.tail, c.ws); err != nil {
		s.ready = 0
		return 0, err
	}
	copy(s.hidden, s.tail[(k-1)*h:])
	s.ready = len(probe)
	p, err := s.next(s.tail[:h], c.enders)
	if err == nil && closed > 0 {
		var after float64
		after, err = s.next(s.tail[closed*h:(closed+1)*h], c.enders)
		p *= after
	}
	if err != nil {
		return 0, err
	}
	s.finished = float32(p)
	return s.finished, nil
}

// next returns the probability that the token after state is one of ids.
func (s *Session) next(state []float32, ids []int) (float64, error) {
	if err := s.c.eval.LogitsInto(state, s.logits, s.c.ws); err != nil {
		return 0, err
	}
	top := s.logits[0]
	for _, v := range s.logits {
		top = max(top, v)
	}
	var sum, in float64
	for _, v := range s.logits {
		sum += math.Exp(float64(v - top))
	}
	for _, id := range ids {
		in += math.Exp(float64(s.logits[id] - top))
	}
	return in / sum, nil
}

// ending reports whether a token is only closing punctuation, perhaps with
// closing quotes and white space, or only white space: text ending with
// either ends.
func ending(piece []byte) (closes, blank bool) {
	if len(piece) == 0 {
		return false, false
	}
	for _, r := range string(piece) {
		switch {
		case r == '.' || r == '?' || r == '!' || r == '…' || r == '。' || r == '？' || r == '！':
			closes = true
		case r == '”' || r == '’' || r == '»' || r == ')' || unicode.IsSpace(r):
		default:
			return false, false
		}
	}
	return closes, !closes
}

// Prefill evaluates the conversation so far.
func (s *Session) Prefill(ctx context.Context) error {
	if s.closed {
		return chat.ErrClosed
	}
	c := s.c
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return chat.ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	limit := c.weights.Config().MaxPositions
	if len(s.ids)+len(c.answer)+2 > limit {
		return errors.New("qwen3: the conversation fills the context")
	}
	// Room for the reply that follows, as Reply reserves it.
	if err := s.reserve(min(len(s.ids)+len(c.answer)+256+2, limit), limit); err != nil {
		return err
	}
	keep := s.kv.CommonPrefix(s.ids)
	if keep == len(s.ids) {
		return nil
	}
	s.ready = 0
	return c.eval.HiddenLastExtendInto(s.kv, keep, s.ids[keep:], s.hidden, c.ws)
}

// emit writes the complete UTF-8 prefix of the text decoded so far, plus
// piece, to w, keeping an incomplete trailing sequence for the next piece;
// flush writes everything.
func (s *Session) emit(piece []byte, w io.Writer, flush bool) error {
	s.text = append(s.text, piece...)
	cut := len(s.text)
	if !flush {
		cut = completeUTF8(s.text)
	}
	if cut == 0 {
		return nil
	}
	_, err := w.Write(s.text[:cut])
	s.text = s.text[:copy(s.text, s.text[cut:])]
	return err
}

// completeUTF8 returns the length of b without a trailing incomplete UTF-8
// sequence.
func completeUTF8(b []byte) int {
	for back := 1; back <= min(3, len(b)); back++ {
		r := b[len(b)-back]
		if r < utf8.RuneSelf {
			return len(b)
		}
		if utf8.RuneStart(r) {
			if !utf8.FullRune(b[len(b)-back:]) {
				return len(b) - back
			}
			return len(b)
		}
	}
	return len(b)
}

// Truncate shortens the last reply to its first n bytes.
func (s *Session) Truncate(n int) error {
	if s.closed {
		return chat.ErrClosed
	}
	if s.reply < 0 {
		return errors.New("qwen3: no reply to truncate")
	}
	c := s.c
	end := len(s.ids) - 1 - len(c.newline) // the reply's <|im_end|>
	keep, size := s.reply, 0
	for keep < end {
		size += len(c.tokens.Piece(s.ids[keep]))
		if size > n {
			break
		}
		keep++
	}
	s.ids = append(s.ids[:keep], c.imEnd)
	s.ids = append(s.ids, c.newline...)
	return nil
}

// Checkpoint marks the conversation.
func (s *Session) Checkpoint() int { return len(s.ids) }

// Restore returns the conversation to a mark from Checkpoint.
func (s *Session) Restore(mark int) error {
	if s.closed {
		return chat.ErrClosed
	}
	if mark < 0 || mark > len(s.ids) {
		return fmt.Errorf("qwen3: mark %d outside the conversation's %d tokens", mark, len(s.ids))
	}
	s.ids = s.ids[:mark]
	s.reply = -1
	return nil
}

// reserve makes the key and value store hold need tokens, growing it by
// doubling up to limit.
func (s *Session) reserve(need, limit int) error {
	if s.kv != nil && s.kv.Capacity() >= need {
		return nil
	}
	capacity := 512
	for capacity < need {
		capacity *= 2
	}
	kv, err := s.c.eval.NewPrefixKV(min(capacity, limit))
	if err != nil {
		return err
	}
	if s.kv != nil {
		n := s.kv.CommonPrefix(s.ids)
		kv.CopyPrefix(s.kv, n)
	}
	s.kv = kv
	return nil
}

// Close ends the session.
func (s *Session) Close() error {
	s.closed = true
	s.kv, s.ids = nil, nil
	return nil
}

// sampler draws tokens from logits; its scratch is reused across steps.
type sampler struct {
	rng  *rand.Rand
	pcg  rand.PCG
	idx  []int32
	prob []float32
}

// candidates bounds nucleus sampling without TopK: tokens beyond the 1024
// most likely carry no measurable probability after a temperature of at
// most one.
const candidates = 1024

func (s *sampler) reset(opts chat.Options) {
	s.pcg.Seed(opts.Seed, opts.Seed^0x9e3779b97f4a7c15)
	if s.rng == nil {
		s.rng = rand.New(&s.pcg)
	}
}

func (s *sampler) next(logits []float32, opts chat.Options) int {
	if opts.Temperature <= 0 {
		return argmax(logits)
	}
	k := opts.TopK
	if k <= 0 || k > candidates {
		k = candidates
	}
	k = min(k, len(logits))
	s.topK(logits, k)
	// Softmax over the candidates, most likely first.
	inv := 1 / opts.Temperature
	top := s.prob[0]
	var sum float32
	for i := range k {
		p := float32(math.Exp(float64((s.prob[i] - top) * inv)))
		s.prob[i] = p
		sum += p
	}
	n := k
	if p := opts.TopP; p > 0 && p < 1 {
		var acc float32
		for i := range k {
			acc += s.prob[i]
			if acc >= p*sum {
				n = i + 1
				sum = acc
				break
			}
		}
	}
	r := s.rng.Float32() * sum
	for i := range n {
		r -= s.prob[i]
		if r <= 0 {
			return int(s.idx[i])
		}
	}
	return int(s.idx[n-1])
}

// topK leaves the k largest logits in s.prob, in descending order, with
// their indices in s.idx.
func (s *sampler) topK(logits []float32, k int) {
	if cap(s.idx) < k {
		s.idx, s.prob = make([]int32, k), make([]float32, k)
	}
	s.idx, s.prob = s.idx[:k], s.prob[:k]
	// A min-heap of the k best seen so far.
	for i := range k {
		s.idx[i], s.prob[i] = int32(i), logits[i]
		for j := i; j > 0; {
			parent := (j - 1) / 2
			if s.prob[parent] <= s.prob[j] {
				break
			}
			s.swap(j, parent)
			j = parent
		}
	}
	for i := k; i < len(logits); i++ {
		if logits[i] <= s.prob[0] {
			continue
		}
		s.idx[0], s.prob[0] = int32(i), logits[i]
		s.down(0, k)
	}
	// Heap sort into descending order.
	for end := k - 1; end > 0; end-- {
		s.swap(0, end)
		s.down(0, end)
	}
}

func (s *sampler) down(j, n int) {
	for {
		l := 2*j + 1
		if l >= n {
			return
		}
		m := l
		if r := l + 1; r < n && s.prob[r] < s.prob[l] {
			m = r
		}
		if s.prob[j] <= s.prob[m] {
			return
		}
		s.swap(j, m)
		j = m
	}
}

func (s *sampler) swap(a, b int) {
	s.idx[a], s.idx[b] = s.idx[b], s.idx[a]
	s.prob[a], s.prob[b] = s.prob[b], s.prob[a]
}

// argmax returns the first index of the largest value.
func argmax(values []float32) int {
	best, at := values[0], 0
	for i, v := range values {
		if v > best {
			best, at = v, i
		}
	}
	return at
}
