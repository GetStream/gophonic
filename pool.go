// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import (
	"errors"
	"reflect"
	"sync"
	"time"
)

// DefaultKeepAlive is how long a Pool keeps an unused model open.
const DefaultKeepAlive = 15 * time.Minute

// A Pool shares models among the goroutines that use them. Acquire opens a
// model the first time one of its lanes is asked for; lanes are reused
// after Release; and a model that has gone unused for the pool's keep-alive
// time is closed, with its lanes, until it is asked for again. Reopening is
// fast: prepared weights are cached on disk and mapped.
//
// A Pool is safe for concurrent use.
type Pool struct {
	opts      Options
	keepAlive time.Duration

	mu     sync.Mutex
	models map[string]*pooled
	closed bool
}

// pooled is one model of a pool, open or opening.
type pooled struct {
	p     *Pool
	path  string
	ready chan struct{} // closed once loading finishes
	model *Model        // set when loading succeeded
	err   error         // set when loading failed

	// Guarded by p.mu:
	busy     int                    // leases out
	idle     map[reflect.Type][]any // released lanes, by lane type
	lastUsed time.Time
	timer    *time.Timer
}

// NewPool returns a pool that opens models with opts and closes each one
// once it has gone unused for keepAlive: DefaultKeepAlive when zero, never
// when negative.
func NewPool(opts Options, keepAlive time.Duration) *Pool {
	if keepAlive == 0 {
		keepAlive = DefaultKeepAlive
	}
	return &Pool{opts: opts, keepAlive: keepAlive, models: map[string]*pooled{}}
}

// A Lease is a lane borrowed from a Pool. Its model stays open until the
// lease is released.
type Lease[T any] struct {
	Lane T
	m    *pooled
}

// Model returns the model the lane belongs to.
func (l Lease[T]) Model() *Model { return l.m.model }

// Path returns the path the model was acquired by.
func (l Lease[T]) Path() string { return l.m.path }

// Release returns the lane to its pool for reuse. Call it exactly once, and
// do not use the lane afterwards.
func (l Lease[T]) Release() {
	l.m.p.release(l.m, reflect.TypeFor[T](), l.Lane)
}

// ErrPoolClosed is returned by Acquire after Close.
var ErrPoolClosed = errors.New("gophonic: pool closed")

// Acquire borrows a lane of type T of the model at path, opening the model
// with Open when the pool does not hold it. Concurrent Acquires of a model
// that is opening wait for it. It fails with speech.ErrUnsupported when the
// model does not provide T. Warm calls, which reuse a released lane,
// allocate nothing.
func Acquire[T any](p *Pool, path string) (Lease[T], error) {
	typ := reflect.TypeFor[T]()
	for {
		m, err := p.get(path)
		if err != nil {
			return Lease[T]{}, err
		}
		<-m.ready
		if m.err != nil {
			return Lease[T]{}, m.err
		}
		p.mu.Lock()
		if p.models[path] != m { // closed while this caller waited
			p.mu.Unlock()
			continue
		}
		m.busy++
		m.timer.Stop()
		lanes := m.idle[typ]
		if n := len(lanes); n > 0 {
			lane := lanes[n-1].(T)
			lanes[n-1] = nil
			m.idle[typ] = lanes[:n-1]
			p.mu.Unlock()
			return Lease[T]{lane, m}, nil
		}
		p.mu.Unlock()
		lane, err := Lane[T](m.model)
		if err != nil {
			p.release(m, typ, nil)
			return Lease[T]{}, err
		}
		return Lease[T]{lane, m}, nil
	}
}

// get returns the pool's entry for path, starting to open it if absent.
func (p *Pool) get(path string) (*pooled, error) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, ErrPoolClosed
	}
	if m := p.models[path]; m != nil {
		p.mu.Unlock()
		return m, nil
	}
	m := &pooled{p: p, path: path, ready: make(chan struct{}), idle: map[reflect.Type][]any{}}
	m.timer = time.AfterFunc(time.Hour, func() { p.expire(m) })
	m.timer.Stop()
	p.models[path] = m
	p.mu.Unlock()

	model, err := Open(path, p.opts)
	p.mu.Lock()
	held := p.models[path] == m
	if err != nil && held {
		delete(p.models, path) // a later Acquire tries again
	}
	p.mu.Unlock()
	if err == nil && !held { // the pool closed while the model opened
		model.Close()
		model, err = nil, ErrPoolClosed
	}
	m.model, m.err = model, err
	close(m.ready)
	return m, nil
}

// release returns lane, when not nil, and starts the keep-alive countdown
// when the model is no longer in use.
func (p *Pool) release(m *pooled, typ reflect.Type, lane any) {
	p.mu.Lock()
	if lane != nil {
		m.idle[typ] = append(m.idle[typ], lane)
	}
	m.busy--
	if m.busy > 0 {
		p.mu.Unlock()
		return
	}
	m.lastUsed = time.Now()
	if p.closed || p.models[m.path] != m {
		p.mu.Unlock()
		m.close()
		return
	}
	if p.keepAlive > 0 {
		m.timer.Reset(p.keepAlive)
	}
	p.mu.Unlock()
}

// expire closes m if it is still unused.
func (p *Pool) expire(m *pooled) {
	p.mu.Lock()
	if m.busy > 0 || p.models[m.path] != m || time.Since(m.lastUsed) < p.keepAlive {
		p.mu.Unlock()
		return
	}
	delete(p.models, m.path)
	p.mu.Unlock()
	m.close()
}

// close closes m's lanes and then m. The pool no longer holds it, and no
// lease of it is out.
func (m *pooled) close() {
	m.timer.Stop()
	for _, lanes := range m.idle {
		for _, lane := range lanes {
			if c, ok := lane.(interface{ Close() error }); ok {
				c.Close()
			}
		}
	}
	m.idle = nil
	if m.model != nil {
		m.model.Close()
	}
}

// Loaded lists the paths of the models the pool holds open.
func (p *Pool) Loaded() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var paths []string
	for path, m := range p.models {
		select {
		case <-m.ready:
			if m.err == nil {
				paths = append(paths, path)
			}
		default:
		}
	}
	return paths
}

// Close closes every model: unused ones now, the others when their last
// lease is released. Acquire fails afterwards.
func (p *Pool) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	var idle []*pooled
	for path, m := range p.models {
		delete(p.models, path)
		select {
		case <-m.ready:
			if m.busy == 0 {
				idle = append(idle, m)
			}
		default: // opening: it closes when Open returns
		}
	}
	p.mu.Unlock()
	for _, m := range idle {
		m.close()
	}
	return nil
}
