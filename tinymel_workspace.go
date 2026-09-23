// Copyright 2026 The gofloor authors
// SPDX-License-Identifier: BSD-2-Clause

package gofloor

import "runtime"

// TinyMelWorkspace owns reusable preprocessing, quantization, convolution,
// GRU, pooling, and classifier buffers for one prediction lane.
type TinyMelWorkspace struct {
	featureHost Workspace
	quantized   []uint8
	melScratch  []uint8
	convA       []float32
	convB       []float32
	gru         tinyGRUWorkspace
	poolScores  []float32
	pooled      []float32
	headNormed  []float32
	headHidden  []float32
	logit       [1]float32
	workers     []tinyWorker
	convJob     tinyConvJob
	gruJob      tinyGRUJob
	geluJob     []float32
	closed      bool
}

// MaxTinyMelWorkers is the largest supported helper count. A prediction uses
// one additional goroutine on the calling lane.
const MaxTinyMelWorkers = 7

const tinyMaxWorkers = MaxTinyMelWorkers

// NewTinyMelWorkspace allocates the buffers needed by the optional TinyMelNet
// model. Reuse one workspace per concurrent prediction lane.
func NewTinyMelWorkspace() *TinyMelWorkspace {
	return newTinyMelWorkspace(0)
}

// NewTinyMelWorkspaceWithWorkers opts into persistent helper goroutines for
// feature extraction, dense convolution layers, large GELU activations, and the
// two independent GRU directions. workerCount is the requested number of
// helpers; it is capped at seven and at GOMAXPROCS-1. The default constructor
// remains serial and starts no goroutines.
func NewTinyMelWorkspaceWithWorkers(workerCount int) *TinyMelWorkspace {
	if workerCount < 0 {
		workerCount = 0
	}
	if workerCount > tinyMaxWorkers {
		workerCount = tinyMaxWorkers
	}
	maxWorkers := runtime.GOMAXPROCS(0) - 1
	if workerCount > maxWorkers {
		workerCount = maxWorkers
	}
	return newTinyMelWorkspace(workerCount)
}

func newTinyMelWorkspace(workerCount int) *TinyMelWorkspace {
	ws := &TinyMelWorkspace{
		featureHost: Workspace{
			features: make([]float32, tinyMelCount*tinyFrameCount),
			audio:    newAudioWorkspace(),
		},
		quantized:  make([]uint8, 400*tinyStemChannels),
		melScratch: make([]uint8, tinyMelCount*tinyFrameCount),
		convA:      make([]float32, 400*tinyStemChannels),
		convB:      make([]float32, 400*tinyStemChannels),
		gru:        newTinyGRUWorkspace(),
		poolScores: make([]float32, tinySequenceLength),
		pooled:     make([]float32, tinyGRUHidden*2),
		headNormed: make([]float32, tinyGRUHidden*2),
		headHidden: make([]float32, tinyGRUHidden*2),
	}
	ws.workers = newTinyConvWorkers(workerCount, ws)
	return ws
}

// Close marks the workspace unusable and releases no model data. It is safe to
// call Close more than once.
func (ws *TinyMelWorkspace) Close() {
	if ws == nil || ws.closed {
		return
	}
	ws.closed = true
	ws.featureHost.closed = true
	closeTinyConvWorkers(ws.workers)
}

type tinyConvJob struct {
	conv        tinyConv1D
	input       []uint8
	params      tinyQuantParams
	output      []float32
	inputLength int
	gelu        bool
}

type tinyWorkerKind uint8

const (
	tinyWorkerConv tinyWorkerKind = iota
	tinyWorkerGRUDirection
	tinyWorkerGELU
	tinyWorkerFeatureFFT
	tinyWorkerFeatureMel
	tinyWorkerFeatureLog
	tinyWorkerFeatureMelLog
)

type tinyWorker struct {
	start     chan struct{}
	done      chan struct{}
	exited    chan struct{}
	from, to  int
	kind      tinyWorkerKind
	workspace *TinyMelWorkspace
	fftReal   [fftSize]float64
	fftImag   [fftSize]float64
	maxLog    float64
}

type tinyGRUJob struct {
	input, w, r, b, output []float32
	work                   *tinyGRUWorkspace
	direction              int
}

func newTinyConvWorkers(count int, ws *TinyMelWorkspace) []tinyWorker {
	if count <= 0 {
		return nil
	}
	workers := make([]tinyWorker, count)
	for i := range workers {
		w := &workers[i]
		w.start = make(chan struct{}, 1)
		w.done = make(chan struct{}, 1)
		w.exited = make(chan struct{})
		w.workspace = ws
		go w.run()
	}
	return workers
}

func (w *tinyWorker) run() {
	defer close(w.exited)
	for range w.start {
		switch w.kind {
		case tinyWorkerConv:
			job := &w.workspace.convJob
			runTinyConvRange(job.conv, job.input, job.params, job.output, job.inputLength, w.from, w.to)
			if job.gelu {
				tinyGELUInPlaceRange(job.output, w.from*job.conv.outChannels, w.to*job.conv.outChannels)
			}
		case tinyWorkerGRUDirection:
			job := &w.workspace.gruJob
			runTinyGRUDirection(job.input, job.w, job.r, job.b, job.output, job.work.directionScratch(job.direction), job.direction)
		case tinyWorkerGELU:
			tinyGELUInPlaceRange(w.workspace.geluJob, w.from, w.to)
		case tinyWorkerFeatureFFT:
			fftPowerFrameRange(&w.workspace.featureHost.audio, w.from, w.to, &w.fftReal, &w.fftImag)
		case tinyWorkerFeatureMel:
			melRowRange(&w.workspace.featureHost.audio, w.from, w.to)
		case tinyWorkerFeatureLog:
			w.maxLog = logMelRowRange(&w.workspace.featureHost.audio, w.from, w.to)
		case tinyWorkerFeatureMelLog:
			melRowRange(&w.workspace.featureHost.audio, w.from, w.to)
			w.maxLog = logMelRowRange(&w.workspace.featureHost.audio, w.from, w.to)
		}
		w.done <- struct{}{}
	}
}

// computeFeatures16kParallel reuses the model workers for independent FFT
// frames and mel rows. Each FFT worker has private transform scratch, while
// the serial path retains the shared extractor's original scratch arrays.
func (ws *TinyMelWorkspace) computeFeatures16kParallel(audio, output []float32) error {
	if len(audio) != maxWindowSamples || len(output) != melCount*frameCount {
		return errInvalidFeatureBuffer
	}
	if len(ws.workers) == 0 {
		return computeFeatures16k(audio, output, &ws.featureHost.audio)
	}
	w := &ws.featureHost.audio
	prepareFeatures16k(audio, w)
	ws.runFeatureRanges(tinyWorkerFeatureFFT, frameCount)
	maxLog := ws.runFeatureMelLogRows()
	writeFeatures16k(output, w, maxLog)
	return nil
}

func (ws *TinyMelWorkspace) runFeatureLogRows() float64 {
	return ws.runFeatureLogWork(tinyWorkerFeatureLog)
}

// Mel and log operate on the same independent rows. A worker can consume its
// freshly accumulated row without waiting for other rows; only the global
// maximum and final normalization require all rows to have completed.
func (ws *TinyMelWorkspace) runFeatureMelLogRows() float64 {
	return ws.runFeatureLogWork(tinyWorkerFeatureMelLog)
}

func (ws *TinyMelWorkspace) runFeatureLogWork(kind tinyWorkerKind) float64 {
	rows := melCount
	lanes := len(ws.workers) + 1
	chunk := (rows + lanes - 1) / lanes
	active := lanes - 1
	for i := 0; i < active; i++ {
		worker := &ws.workers[i]
		worker.from = i * chunk
		worker.to = worker.from + chunk
		if worker.to > rows {
			worker.to = rows
		}
		worker.kind = kind
		worker.start <- struct{}{}
	}
	if kind == tinyWorkerFeatureMelLog {
		melRowRange(&ws.featureHost.audio, active*chunk, rows)
	}
	maxLog := logMelRowRange(&ws.featureHost.audio, active*chunk, rows)
	for i := 0; i < active; i++ {
		<-ws.workers[i].done
		if ws.workers[i].maxLog > maxLog {
			maxLog = ws.workers[i].maxLog
		}
	}
	return maxLog
}

func (ws *TinyMelWorkspace) runFeatureRanges(kind tinyWorkerKind, rows int) {
	lanes := len(ws.workers) + 1
	if lanes > rows {
		lanes = rows
	}
	chunk := (rows + lanes - 1) / lanes
	active := lanes - 1
	for i := 0; i < active; i++ {
		worker := &ws.workers[i]
		worker.from = i * chunk
		worker.to = worker.from + chunk
		if worker.to > rows {
			worker.to = rows
		}
		worker.kind = kind
		worker.start <- struct{}{}
	}
	from := active * chunk
	if from < rows {
		if kind == tinyWorkerFeatureFFT {
			fftPowerFrameRange(&ws.featureHost.audio, from, rows, &ws.featureHost.audio.fftReal, &ws.featureHost.audio.fftImag)
		} else {
			melRowRange(&ws.featureHost.audio, from, rows)
		}
	}
	for i := 0; i < active; i++ {
		<-ws.workers[i].done
	}
}

func closeTinyConvWorkers(workers []tinyWorker) {
	for i := range workers {
		close(workers[i].start)
	}
	for i := range workers {
		<-workers[i].exited
	}
}

func (ws *TinyMelWorkspace) runGELU(values []float32) {
	if len(ws.workers) == 0 || len(values) < 4096 {
		tinyGELUInPlace(values)
		return
	}
	ws.geluJob = values
	lanes := len(ws.workers) + 1
	if lanes > len(values) {
		lanes = len(values)
	}
	chunk := (len(values) + lanes - 1) / lanes
	active := lanes - 1
	for i := 0; i < active; i++ {
		worker := &ws.workers[i]
		worker.from = i * chunk
		worker.to = worker.from + chunk
		if worker.to > len(values) {
			worker.to = len(values)
		}
		worker.kind = tinyWorkerGELU
		worker.start <- struct{}{}
	}
	from := active * chunk
	if from < len(values) {
		tinyGELUInPlaceRange(values, from, len(values))
	}
	for i := 0; i < active; i++ {
		<-ws.workers[i].done
	}
	ws.geluJob = nil
}

func (ws *TinyMelWorkspace) runConv(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength int) int {
	return ws.runConvWork(conv, input, params, output, inputLength, false)
}

// runConvGELU keeps each output-time range on its owning lane through GELU.
// Quantized convolution input is separate from output, so another worker's
// reads cannot depend on the float output being activated here.
func (ws *TinyMelWorkspace) runConvGELU(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength int) int {
	return ws.runConvWork(conv, input, params, output, inputLength, true)
}

func (ws *TinyMelWorkspace) runConvWork(conv tinyConv1D, input []uint8, params tinyQuantParams, output []float32, inputLength int, gelu bool) int {
	pad := conv.kernel / 2
	outputLength := (inputLength+2*pad-conv.kernel)/conv.stride + 1
	if len(ws.workers) == 0 || outputLength < 16 || conv.groups == conv.inChannels {
		length := runTinyConv(conv, input, params, output, inputLength)
		if gelu {
			tinyGELUInPlaceRange(output, 0, length*conv.outChannels)
		}
		return length
	}

	ws.convJob = tinyConvJob{
		conv: conv, input: input, params: params, output: output,
		inputLength: inputLength, gelu: gelu,
	}
	lanes := len(ws.workers) + 1
	if lanes > outputLength {
		lanes = outputLength
	}
	chunk := (outputLength + lanes - 1) / lanes
	active := lanes - 1
	for i := 0; i < active; i++ {
		w := &ws.workers[i]
		w.from = i * chunk
		w.to = w.from + chunk
		w.kind = tinyWorkerConv
		if w.to > outputLength {
			w.to = outputLength
		}
		w.start <- struct{}{}
	}
	from := active * chunk
	if from < outputLength {
		runTinyConvRange(conv, input, params, output, inputLength, from, outputLength)
		if gelu {
			tinyGELUInPlaceRange(output, from*conv.outChannels, outputLength*conv.outChannels)
		}
	}
	for i := 0; i < active; i++ {
		<-ws.workers[i].done
	}
	return outputLength
}

func (ws *TinyMelWorkspace) runGRU(model *TinyMelModel, input []float32) {
	if len(ws.workers) == 0 || runtime.GOMAXPROCS(0) < 2 {
		runTinyGRU(input, model.gruW, model.gruR, model.gruB, ws.gru.buffer(), &ws.gru)
		return
	}

	worker := &ws.workers[0]
	ws.gruJob = tinyGRUJob{
		input:  input,
		w:      model.gruW[tinyGRUInputStride : 2*tinyGRUInputStride],
		r:      model.gruR[tinyGRURecurrentStride : 2*tinyGRURecurrentStride],
		b:      model.gruB[tinyGRUBiasStride : 2*tinyGRUBiasStride],
		output: ws.gru.buffer(), work: &ws.gru, direction: 1,
	}
	worker.kind = tinyWorkerGRUDirection
	worker.start <- struct{}{}
	runTinyGRUDirection(
		input,
		model.gruW[:tinyGRUInputStride],
		model.gruR[:tinyGRURecurrentStride],
		model.gruB[:tinyGRUBiasStride],
		ws.gru.buffer(), ws.gru.directionScratch(0), 0,
	)
	<-worker.done
}
