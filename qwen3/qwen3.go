// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package qwen3

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/internal/qwen3lm"
	"github.com/GetStream/gophonic/speech"
)

// Model is one loaded Qwen3-family checkpoint (dense, a mixture of experts,
// or the Qwen3.6 hybrid) that serves conversations (chat.Generator),
// questions (speech.ZeroShot, Question, NewContext), and embeddings from one
// copy of its weights.
//
// Each capability is prepared at its first use, so a model pays only for
// what it does. The first session loads the language-model head and a
// workspace to generate with. The first question or embedding prepares a
// workspace, a store of long inputs' keys and values, and an embedding
// cache. Generation and questions have a workspace each, so a question is
// answered while a reply is written; calls of one kind are serialized. Open
// a second Model for a second lane of the same kind.
type Model struct {
	path    string
	opts    Options
	weights *qwen3lm.Weights
	tokens  *Tokenizer

	mu     sync.Mutex
	gen    *generator
	enc    *encoder
	closed bool
}

var (
	_ chat.Generator  = (*Model)(nil)
	_ speech.ZeroShot = (*Model)(nil)
)

// errClosed is returned by a closed Model.
var errClosed = errors.New("qwen3: model closed")

// Open loads an official Qwen3 safetensors snapshot directory, such as
// Qwen/Qwen3-8B, Qwen/Qwen3-30B-A3B-Instruct-2507, or Qwen/Qwen3.6-35B-A3B.
// The zero Options value selects the fastest faithful weight format and
// the default caches.
func Open(path string, opts Options) (*Model, error) {
	if opts.Threads < 0 {
		return nil, fmt.Errorf("qwen3: invalid thread count %d", opts.Threads)
	}
	tokens, err := LoadTokenizer(path)
	if err != nil {
		return nil, fmt.Errorf("qwen3: load tokenizer: %w", err)
	}
	weights, err := qwen3lm.LoadWeights(path, opts.Format)
	if err != nil {
		return nil, err
	}
	return &Model{path: path, opts: opts, weights: weights, tokens: tokens}, nil
}

// generator returns the model's generator, preparing it at first use.
func (m *Model) generator() (*generator, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	switch {
	case m.closed:
		return nil, errClosed
	case m.gen != nil:
		return m.gen, nil
	}
	head, err := headOf(m.path)
	if err != nil {
		return nil, err
	}
	if err := m.weights.LoadHead(head); err != nil {
		return nil, err
	}
	if m.gen, err = newGenerator(m.weights, m.tokens, m.opts.threads(), m.path); err != nil {
		return nil, err
	}
	return m.gen, nil
}

// encoder returns the model's encoder, preparing it at first use.
func (m *Model) encoder() (*encoder, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	switch {
	case m.closed:
		return nil, errClosed
	case m.enc != nil:
		return m.enc, nil
	}
	var err error
	m.enc, err = newEncoder(m.weights, m.tokens, m.path, m.opts)
	return m.enc, err
}

// NewSession starts a conversation with system as its system prompt, and
// tools as the functions the model may call; see chat.Generator.
func (m *Model) NewSession(system string, tools ...chat.ToolSpec) (chat.Session, error) {
	g, err := m.generator()
	if err != nil {
		return nil, err
	}
	return g.NewSession(system, tools...)
}

// Classifier prepares a multiple-choice question as a classifier of text;
// see speech.ZeroShot.
func (m *Model) Classifier(question string, labels []string) (speech.TextClassifier, error) {
	e, err := m.encoder()
	if err != nil {
		return nil, err
	}
	return e.Classifier(question, labels)
}

// Question prepares a multiple-choice question about texts: each Choose
// is one prefill and no text generation.
func (m *Model) Question(question string, options []string) (*Question, error) {
	e, err := m.encoder()
	if err != nil {
		return nil, err
	}
	return e.Question(question, options)
}

// NewContext prepares a context of at most maxTokens tokens that any
// number of ContextQuestions are asked about.
func (m *Model) NewContext(maxTokens int) (*Context, error) {
	e, err := m.encoder()
	if err != nil {
		return nil, err
	}
	return e.NewContext(maxTokens)
}

// ContextQuestion prepares a question about a Context.
func (m *Model) ContextQuestion(question string, options []string) (*ContextQuestion, error) {
	e, err := m.encoder()
	if err != nil {
		return nil, err
	}
	return e.ContextQuestion(question, options)
}

// Embed writes each text's raw, post-final-norm, last-token hidden state
// (the model's width: 4096 values for Qwen3-8B) to the matching dst row,
// keeping the last 2048 tokens of longer texts as the CLM reference does.
// No special tokens are added. Texts of one call are packed into shared
// forward passes. Once warm to the call's shape, it allocates nothing.
//
// Embed has the shape of a clm.Embedder without its role argument; the
// published CLM heads use the same encoding for states and actions:
//
//	clm.EmbedFunc(func(ctx context.Context, _ clm.Role, texts []string, dst [][]float32) error {
//		return m.Embed(ctx, texts, dst)
//	})
func (m *Model) Embed(ctx context.Context, texts []string, dst [][]float32) error {
	e, err := m.encoder()
	if err != nil {
		return err
	}
	return e.Embed(ctx, texts, dst)
}

// EmbedTokensInto embeds caller-tokenized inputs into caller-owned vectors,
// as Embed does texts.
func (m *Model) EmbedTokensInto(ctx context.Context, ids [][]int, dst [][]float32) error {
	e, err := m.encoder()
	if err != nil {
		return err
	}
	return e.EmbedTokensInto(ctx, ids, dst)
}

// Width reports the length of an embedding: the model's hidden size, 4096
// for Qwen3-8B.
func (m *Model) Width() int { return m.weights.Config().Hidden }

// PrefixStats reports, for inputs of at least 64 tokens, how many tokens
// were served from the prefix store and how many were evaluated.
func (m *Model) PrefixStats() (reused, computed uint64) {
	m.mu.Lock()
	e := m.enc
	m.mu.Unlock()
	return e.PrefixStats()
}

// CacheStats reports embedding-cache hits and lookups since Open.
func (m *Model) CacheStats() (hits, lookups uint64) {
	m.mu.Lock()
	e := m.enc
	m.mu.Unlock()
	return e.CacheStats()
}

// Close releases the model once its sessions, questions, and contexts are
// done: its workspaces, and its weights' memory. It is safe to call more
// than once.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	var err error
	if m.gen != nil {
		err = m.gen.Close()
	}
	if m.enc != nil {
		err = errors.Join(err, m.enc.Close())
	}
	m.weights.Release()
	return err
}
