// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package smartturn

import (
	"runtime"

	"github.com/GetStream/gophonic/internal/mel"
)

// Workspace owns all scratch buffers used by prediction. Reuse one workspace
// per concurrent goroutine to avoid allocations in the inference path.
type Workspace struct {
	features              []float32
	audio                 *mel.Turn
	conv1Windows          []float32
	conv1                 []float32
	conv2Windows          []float32
	q, k, v, attentionOut []float32
	hiddenA, hiddenB      []float32
	normed                []float32
	attention             []float32
	ffn                   []float32
	poolHidden            []float32
	poolScores            []float32
	pooled                []float32
	classHidden           []float32
	classMid              []float32
	classLogit            []float32
	workers               []worker
	job                   parallelJob
	closed                bool
}

// NewWorkspace allocates the reusable scratch buffers for one prediction lane.
func NewWorkspace() *Workspace {
	w := &Workspace{
		features:     make([]float32, melCount*frameCount),
		audio:        mel.NewTurn(),
		conv1Windows: make([]float32, frameCount*melCount*3),
		conv1:        make([]float32, hiddenSize*frameCount),
		conv2Windows: make([]float32, sequenceLength*hiddenSize*3),
		q:            make([]float32, sequenceLength*hiddenSize),
		k:            make([]float32, sequenceLength*hiddenSize),
		v:            make([]float32, sequenceLength*hiddenSize),
		attentionOut: make([]float32, sequenceLength*hiddenSize),
		hiddenA:      make([]float32, sequenceLength*hiddenSize),
		hiddenB:      make([]float32, sequenceLength*hiddenSize),
		normed:       make([]float32, sequenceLength*hiddenSize),
		attention:    make([]float32, attentionHeads*sequenceLength*sequenceLength),
		ffn:          make([]float32, sequenceLength*feedForwardSize),
		poolHidden:   make([]float32, sequenceLength*256),
		poolScores:   make([]float32, sequenceLength),
		pooled:       make([]float32, hiddenSize),
		classHidden:  make([]float32, 256),
		classMid:     make([]float32, 64),
		classLogit:   make([]float32, 1),
	}
	workerCount := runtime.GOMAXPROCS(0) - 1
	if workerCount > 0 {
		w.workers = make([]worker, workerCount)
		for i := range w.workers {
			w.workers[i] = worker{start: make(chan struct{}, 1), done: make(chan struct{}, 1), exited: make(chan struct{})}
			go w.workers[i].run(w, i)
		}
	}
	return w
}

// Close stops the workspace's worker goroutines. A closed workspace cannot be reused.
func (w *Workspace) Close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	for i := range w.workers {
		close(w.workers[i].start)
	}
	for i := range w.workers {
		<-w.workers[i].exited
	}
}
