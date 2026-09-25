// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package httpserver

import (
	"slices"
	"strings"
	"sync"

	"github.com/GetStream/gophonic"
	"github.com/GetStream/gophonic/speech"
)

// maxQuestions bounds the prepared zero-shot classifiers the server keeps.
// A prepared question holds its prompt's keys and values: tens of MiB.
const maxQuestions = 16

// questions caches prepared zero-shot classifiers by model, question, and
// labels, most recently used last. Lookups allocate nothing.
type questions struct {
	mu      sync.Mutex
	entries []*question
}

// question is one prepared classifier. Its mutex, held while in use,
// serializes the requests that share it.
type question struct {
	mu       sync.Mutex
	model    *gophonic.Model
	path     string
	question string
	labels   []string
	c        speech.TextClassifier
}

// get returns the classifier for question and labels on model, which the
// pool holds as path, preparing it with z if needed. It returns the entry
// locked; the caller unlocks it when done.
func (qs *questions) get(model *gophonic.Model, path string, z speech.ZeroShot, text string, labels []string) (*question, error) {
	qs.mu.Lock()
	for i, e := range qs.entries {
		if e.model == model && e.question == text && slices.Equal(e.labels, labels) {
			copy(qs.entries[i:], qs.entries[i+1:])
			qs.entries[len(qs.entries)-1] = e
			qs.mu.Unlock()
			e.mu.Lock()
			return e, nil
		}
	}
	qs.mu.Unlock()

	// The request's strings alias its body; the cache keeps copies.
	e := &question{model: model, path: path, question: strings.Clone(text), labels: make([]string, len(labels))}
	for i, l := range labels {
		e.labels[i] = strings.Clone(l)
	}
	c, err := z.Classifier(e.question, e.labels)
	if err != nil {
		return nil, err
	}
	e.c = c
	e.mu.Lock()

	var evict []*question
	qs.mu.Lock()
	// Questions of the same path prepared on an earlier opening of the
	// model are dead: the pool closed it.
	qs.entries = slices.DeleteFunc(qs.entries, func(o *question) bool {
		stale := o.path == path && o.model != model
		if stale {
			evict = append(evict, o)
		}
		return stale
	})
	qs.entries = append(qs.entries, e)
	if n := len(qs.entries) - maxQuestions; n > 0 {
		evict = append(evict, qs.entries[:n]...)
		qs.entries = slices.Delete(qs.entries, 0, n)
	}
	qs.mu.Unlock()
	for _, o := range evict {
		o.mu.Lock() // wait for a request still using it
		o.c.Close()
		o.mu.Unlock()
	}
	return e, nil
}

// close closes every prepared classifier.
func (qs *questions) close() {
	qs.mu.Lock()
	entries := qs.entries
	qs.entries = nil
	qs.mu.Unlock()
	for _, e := range entries {
		e.mu.Lock()
		e.c.Close()
		e.mu.Unlock()
	}
}
