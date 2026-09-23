// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gophonic

import "math"

type workKind uint8

const (
	workLinear workKind = iota
	workQKV
	workLayerNorm
	workAttention
	workConv1
	workConv2
)

type parallelJob struct {
	kind                      workKind
	rows, width, in, out      int
	input, output             []float32
	weights, bias             []float32
	gamma, beta               []float32
	positions                 []float32
	layer                     linear
	qLayer, kLayer, vLayer    linear
	qOutput, kOutput, vOutput []float32
	tileRows                  int
	epsilon                   float32
	q, k, v, scores           []float32
}

type worker struct {
	start    chan struct{}
	done     chan struct{}
	exited   chan struct{}
	from, to int
}

func (w *worker) run(ws *Workspace, _ int) {
	defer close(w.exited)
	for range w.start {
		runWorkRange(&ws.job, w.from, w.to)
		w.done <- struct{}{}
	}
}

func (ws *Workspace) runRows(rows int) {
	if rows <= 0 {
		return
	}
	if ws.closed || len(ws.workers) == 0 || rows < 16 {
		runWorkRange(&ws.job, 0, rows)
		return
	}
	workerCount := len(ws.workers) + 1
	if workerCount > rows {
		workerCount = rows
	}
	chunk := (rows + workerCount - 1) / workerCount
	active := workerCount - 1
	for i := 0; i < active; i++ {
		w := &ws.workers[i]
		w.from = i * chunk
		w.to = w.from + chunk
		if w.to > rows {
			w.to = rows
		}
		w.start <- struct{}{}
	}
	from := active * chunk
	if from < rows {
		runWorkRange(&ws.job, from, rows)
	}
	for i := 0; i < active; i++ {
		<-ws.workers[i].done
	}
}

func runWorkRange(job *parallelJob, from, to int) {
	switch job.kind {
	case workLinear:
		if job.tileRows != 0 {
			runLinearTiles(job, from, to)
			return
		}
		for r := from; r < to; r++ {
			x := job.input[r*job.in : (r+1)*job.in]
			dst := job.output[r*job.out : (r+1)*job.out]
			o := 0
			for ; o+4 <= job.out; o += 4 {
				s0, s1, s2, s3 := dotProduct4(
					x,
					job.layer.w[o*job.in:(o+1)*job.in],
					job.layer.w[(o+1)*job.in:(o+2)*job.in],
					job.layer.w[(o+2)*job.in:(o+3)*job.in],
					job.layer.w[(o+3)*job.in:(o+4)*job.in],
				)
				if job.layer.b != nil {
					s0 += job.layer.b[o]
					s1 += job.layer.b[o+1]
					s2 += job.layer.b[o+2]
					s3 += job.layer.b[o+3]
				}
				dst[o], dst[o+1], dst[o+2], dst[o+3] = s0, s1, s2, s3
			}
			for ; o < job.out; o++ {
				sum := dotProduct(x, job.layer.w[o*job.in:(o+1)*job.in])
				if job.layer.b != nil {
					sum += job.layer.b[o]
				}
				dst[o] = sum
			}
		}
	case workQKV:
		runQKVTiles(job, from, to)
	case workLayerNorm:
		runLayerNormRange(job, from, to)
	case workAttention:
		runAttentionRange(job, from, to)
	case workConv1:
		const width = melCount * 3
		for oc := from; oc < to; oc++ {
			weight := job.weights[oc*width : (oc+1)*width]
			for t := 0; t < frameCount; t++ {
				sum := dotProduct(weight, job.input[t*width:(t+1)*width]) + job.bias[oc]
				job.output[oc*frameCount+t] = gelu(sum)
			}
		}
	case workConv2:
		const width = hiddenSize * 3
		for oc := from; oc < to; oc++ {
			weight := job.weights[oc*width : (oc+1)*width]
			for t := 0; t < sequenceLength; t++ {
				sum := dotProduct(weight, job.input[t*width:(t+1)*width]) + job.bias[oc]
				index := t*hiddenSize + oc
				job.output[index] = gelu(sum) + job.positions[index]
			}
		}
	}
}

func linearInto(ws *Workspace, input []float32, rows int, layer linear, output []float32, in, out int) {
	tileRows := 0
	workRows := rows
	if rows >= 16 && out >= 4 && out%4 == 0 {
		tileRows = 4
		workRows = (rows + tileRows - 1) / tileRows
	}
	ws.job = parallelJob{kind: workLinear, rows: rows, in: in, out: out, input: input, output: output, layer: layer, tileRows: tileRows}
	ws.runRows(workRows)
}

func runLinearTiles(job *parallelJob, from, to int) {
	for group := from; group < to; group++ {
		firstRow := group * 4
		if firstRow+4 > job.rows {
			for r := firstRow; r < job.rows; r++ {
				runLinearRow(job, r)
			}
			continue
		}
		for o := 0; o+4 <= job.out; o += 4 {
			inputBase := firstRow * job.in
			writeLinearTile4x4(job, inputBase, job.layer, job.output, job.out, o, firstRow)
		}
	}
}

func runQKVTiles(job *parallelJob, from, to int) {
	for group := from; group < to; group++ {
		firstRow := group * 4
		if firstRow+4 > job.rows {
			for r := firstRow; r < job.rows; r++ {
				input := job.input[r*job.in : (r+1)*job.in]
				if job.gamma != nil {
					var normalized [hiddenSize]float32
					normJob := parallelJob{
						width: job.in, input: input, output: normalized[:],
						gamma: job.gamma, beta: job.beta, epsilon: job.epsilon,
					}
					runLayerNormRange(&normJob, 0, 1)
					input = normalized[:]
				}
				qJob := parallelJob{in: job.in, out: job.out, input: input, output: job.qOutput[r*job.out:], layer: job.qLayer}
				kJob := parallelJob{in: job.in, out: job.out, input: input, output: job.kOutput[r*job.out:], layer: job.kLayer}
				vJob := parallelJob{in: job.in, out: job.out, input: input, output: job.vOutput[r*job.out:], layer: job.vLayer}
				runLinearRow(&qJob, 0)
				runLinearRow(&kJob, 0)
				runLinearRow(&vJob, 0)
			}
			continue
		}
		var normalized [4 * hiddenSize]float32
		input := job.input[firstRow*job.in : (firstRow+4)*job.in]
		if job.gamma != nil {
			normJob := parallelJob{
				width: job.in, rows: 4, input: input, output: normalized[:],
				gamma: job.gamma, beta: job.beta, epsilon: job.epsilon,
			}
			runLayerNormRange(&normJob, 0, 4)
			input = normalized[:]
		}
		x0 := input[:job.in]
		x1 := input[job.in : 2*job.in]
		x2 := input[2*job.in : 3*job.in]
		x3 := input[3*job.in : 4*job.in]
		for o := 0; o+4 <= job.out; o += 4 {
			writeLinearTile4x4Rows(x0, x1, x2, x3, job.qLayer, job.qOutput, job.out, o, firstRow)
			writeLinearTile4x4Rows(x0, x1, x2, x3, job.kLayer, job.kOutput, job.out, o, firstRow)
			writeLinearTile4x4Rows(x0, x1, x2, x3, job.vLayer, job.vOutput, job.out, o, firstRow)
		}
	}
}

func writeLinearTile4x4(job *parallelJob, inputBase int, layer linear, output []float32, outWidth, outputStart, firstRow int) {
	writeLinearTile4x4Rows(
		job.input[inputBase:inputBase+job.in],
		job.input[inputBase+job.in:inputBase+2*job.in],
		job.input[inputBase+2*job.in:inputBase+3*job.in],
		job.input[inputBase+3*job.in:inputBase+4*job.in],
		layer, output, outWidth, outputStart, firstRow,
	)
}

func writeLinearTile4x4Rows(x0, x1, x2, x3 []float32, layer linear, output []float32, outWidth, outputStart, firstRow int) {
	var sums [16]float32
	inWidth := len(x0)
	dotProduct4x4(
		x0, x1, x2, x3,
		layer.w[outputStart*inWidth:(outputStart+1)*inWidth],
		layer.w[(outputStart+1)*inWidth:(outputStart+2)*inWidth],
		layer.w[(outputStart+2)*inWidth:(outputStart+3)*inWidth],
		layer.w[(outputStart+3)*inWidth:(outputStart+4)*inWidth],
		&sums,
	)
	for r := 0; r < 4; r++ {
		base := (firstRow+r)*outWidth + outputStart
		for c := 0; c < 4; c++ {
			sum := sums[r*4+c]
			if layer.b != nil {
				sum += layer.b[outputStart+c]
			}
			output[base+c] = sum
		}
	}
}

func linearQKVInto(ws *Workspace, input []float32, rows int, q, k, v linear, qOutput, kOutput, vOutput []float32, width int, gamma, beta []float32, epsilon float32) {
	workRows := (rows + 3) / 4
	ws.job = parallelJob{
		kind: workQKV, rows: rows, in: width, out: width, input: input,
		qLayer: q, kLayer: k, vLayer: v, qOutput: qOutput, kOutput: kOutput, vOutput: vOutput,
		tileRows: 4, gamma: gamma, beta: beta, epsilon: epsilon,
	}
	ws.runRows(workRows)
}

func runLinearRow(job *parallelJob, r int) {
	x := job.input[r*job.in : (r+1)*job.in]
	dst := job.output[r*job.out : (r+1)*job.out]
	for o := 0; o < job.out; o++ {
		sum := dotProduct(x, job.layer.w[o*job.in:(o+1)*job.in])
		if job.layer.b != nil {
			sum += job.layer.b[o]
		}
		dst[o] = sum
	}
}

func layerNormInto(ws *Workspace, input, output []float32, rows, width int, gamma, beta []float32, epsilon float32) {
	ws.job = parallelJob{kind: workLayerNorm, rows: rows, width: width, input: input, output: output, gamma: gamma, beta: beta, epsilon: epsilon}
	ws.runRows(rows)
}

func layerNormInPlace(ws *Workspace, values []float32, rows, width int, gamma, beta []float32, epsilon float32) {
	layerNormInto(ws, values, values, rows, width, gamma, beta, epsilon)
}

func runLayerNormRange(job *parallelJob, from, to int) {
	for r := from; r < to; r++ {
		row := job.input[r*job.width : (r+1)*job.width]
		dst := job.output[r*job.width : (r+1)*job.width]
		var sum float32
		for _, x := range row {
			sum += x
		}
		mean := sum / float32(job.width)
		var variance float32
		for _, x := range row {
			d := x - mean
			variance += d * d
		}
		variance /= float32(job.width)
		inv := 1 / float32(math.Sqrt(float64(variance+job.epsilon)))
		for i, x := range row {
			dst[i] = (x-mean)*inv*job.gamma[i] + job.beta[i]
		}
	}
}

func runAttentionRange(job *parallelJob, from, to int) {
	const queryBlocks = sequenceLength / 4
	const scale float32 = 0.125
	for block := from; block < to; block++ {
		head := block / queryBlocks
		firstQuery := (block % queryBlocks) * 4
		headOffset := head * headSize
		q0 := job.q[firstQuery*hiddenSize+headOffset : firstQuery*hiddenSize+headOffset+headSize]
		q1 := job.q[(firstQuery+1)*hiddenSize+headOffset : (firstQuery+1)*hiddenSize+headOffset+headSize]
		q2 := job.q[(firstQuery+2)*hiddenSize+headOffset : (firstQuery+2)*hiddenSize+headOffset+headSize]
		q3 := job.q[(firstQuery+3)*hiddenSize+headOffset : (firstQuery+3)*hiddenSize+headOffset+headSize]
		for firstKey := 0; firstKey < sequenceLength; firstKey += 4 {
			k0 := job.k[firstKey*hiddenSize+headOffset : firstKey*hiddenSize+headOffset+headSize]
			k1 := job.k[(firstKey+1)*hiddenSize+headOffset : (firstKey+1)*hiddenSize+headOffset+headSize]
			k2 := job.k[(firstKey+2)*hiddenSize+headOffset : (firstKey+2)*hiddenSize+headOffset+headSize]
			k3 := job.k[(firstKey+3)*hiddenSize+headOffset : (firstKey+3)*hiddenSize+headOffset+headSize]
			var tile [16]float32
			dotProduct4x4(q0, q1, q2, q3, k0, k1, k2, k3, &tile)
			for r := 0; r < 4; r++ {
				row := ((firstQuery+r)*attentionHeads + head) * sequenceLength
				for c := 0; c < 4; c++ {
					job.scores[row+firstKey+c] = tile[r*4+c] * scale
				}
			}
		}
		row0 := (firstQuery*attentionHeads + head) * sequenceLength
		row1 := ((firstQuery+1)*attentionHeads + head) * sequenceLength
		row2 := ((firstQuery+2)*attentionHeads + head) * sequenceLength
		row3 := ((firstQuery+3)*attentionHeads + head) * sequenceLength
		scores0 := job.scores[row0 : row0+sequenceLength]
		scores1 := job.scores[row1 : row1+sequenceLength]
		scores2 := job.scores[row2 : row2+sequenceLength]
		scores3 := job.scores[row3 : row3+sequenceLength]
		softmaxInPlace(scores0)
		softmaxInPlace(scores1)
		softmaxInPlace(scores2)
		softmaxInPlace(scores3)
		output0 := job.output[firstQuery*hiddenSize+headOffset:]
		output1 := job.output[(firstQuery+1)*hiddenSize+headOffset:]
		output2 := job.output[(firstQuery+2)*hiddenSize+headOffset:]
		output3 := job.output[(firstQuery+3)*hiddenSize+headOffset:]
		attentionContext4(scores0, scores1, scores2, scores3, job.v, output0, output1, output2, output3, headOffset)
	}
}
