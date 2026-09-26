// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package gpuportable

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/gogpu/wgpu"
)

// LinearFormat selects the on-device representation for a row-major linear
// projection. Both formats keep their checkpoint/quantized weights packed.
type LinearFormat uint8

const (
	LinearBF16 LinearFormat = iota + 1
	LinearQ8B
)

const (
	linearParamStride = 256
)

// Linear is an immutable row-major matrix and its shared projection pipeline.
// Matrix data is uploaded in bounded chunks and split into storage bindings
// when the adapter's actual per-binding limit requires it.
type Linear struct {
	engine           *Engine
	format           LinearFormat
	rows             int
	cols             int
	workgroupSize    uint32
	rowsPerWorkgroup uint32
	vectorized       bool
	chunks           []linearChunk
	kernel           *Kernel
	closed           bool
}

type linearChunk struct {
	row0   int
	rows   int
	weight *wgpu.Buffer
	scale  *wgpu.Buffer
}

// NewLinear creates a projection from already packed bytes. LinearBF16 uses
// little-endian BF16 values in row-major order and requires an even column
// count. LinearQ8B uses row-major signed int8 codes (one byte per value) and
// row-major FP16 scales (one value for each block of 32 columns).
func NewLinear(e *Engine, format LinearFormat, rows, cols int, weights, scales []byte, uploadChunkBytes int) (*Linear, error) {
	workgroupSize, rowsPerWorkgroup, err := defaultLinearGeometry(e, cols)
	if err != nil {
		return nil, err
	}
	return newLinearGeometry(e, format, rows, cols, weights, scales, uploadChunkBytes, workgroupSize, rowsPerWorkgroup)
}

func newLinearGeometry(e *Engine, format LinearFormat, rows, cols int, weights, scales []byte, uploadChunkBytes int, workgroupSize, rowsPerWorkgroup uint32) (_ *Linear, err error) {
	if e == nil || e.device == nil || rows <= 0 || cols <= 0 {
		return nil, errors.New("gpuportable: invalid linear dimensions or engine")
	}
	maxInt := int(^uint(0) >> 1)
	if uint64(rows) > uint64(^uint32(0)) || uint64(cols) > uint64(^uint32(0)) || cols > maxInt/4 || rows > maxInt/4 {
		return nil, errors.New("gpuportable: linear dimensions exceed supported indexing")
	}
	var weightRowBytes, scaleRowBytes int
	switch format {
	case LinearBF16:
		if cols%2 != 0 {
			return nil, errors.New("gpuportable: BF16 linear column count must be even")
		}
		weightRowBytes = cols * 2
		if len(scales) != 0 {
			return nil, errors.New("gpuportable: BF16 linear does not use a scale buffer")
		}
	case LinearQ8B:
		if cols%32 != 0 {
			return nil, errors.New("gpuportable: Q8B linear column count must be a multiple of 32")
		}
		weightRowBytes = cols
		scaleRowBytes = (cols / 32) * 2
	default:
		return nil, fmt.Errorf("gpuportable: unsupported linear format %d", format)
	}
	if rows > maxInt/weightRowBytes || len(weights) != rows*weightRowBytes {
		return nil, errors.New("gpuportable: packed weight length does not match linear dimensions")
	}
	if format == LinearQ8B && (rows > maxInt/scaleRowBytes || len(scales) != rows*scaleRowBytes) {
		return nil, errors.New("gpuportable: Q8B scale length does not match linear dimensions")
	}
	limits := e.Limits()
	maxBinding := limits.MaxStorageBufferBindingSize
	if maxBinding == 0 || uint64(cols)*4 > maxBinding || uint64(rows)*4 > maxBinding {
		return nil, fmt.Errorf("gpuportable: activation storage exceeds adapter binding limit %d", maxBinding)
	}
	maxGroups := int(limits.MaxComputeWorkgroupsPerDimension)
	if maxGroups == 0 {
		return nil, errors.New("gpuportable: adapter reported zero compute workgroups per dimension")
	}
	vectorized := format == LinearQ8B || cols%16 == 0
	if !vectorized && rowsPerWorkgroup != 1 {
		return nil, errors.New("gpuportable: unaligned BF16 linear supports one row per workgroup")
	}
	tilesPerWorkgroup := uint64(1)
	if vectorized {
		tilesPerWorkgroup = uint64(workgroupSize / linearLogicalTileSize)
	}
	rowsPerGroup := tilesPerWorkgroup * uint64(rowsPerWorkgroup)
	maxRows64 := min(uint64(rows), uint64(maxGroups)*rowsPerGroup, maxBinding/uint64(weightRowBytes))
	maxRows64 = min(maxRows64, (uint64(^uint32(0))+1)/uint64(cols))
	if format == LinearQ8B {
		maxRows64 = min(maxRows64, maxBinding/uint64(scaleRowBytes))
	}
	maxRows := int(maxRows64)
	if maxRows < 1 {
		return nil, errors.New("gpuportable: one projection row exceeds adapter storage limits")
	}
	kernel, err := e.linearKernel(format, workgroupSize, rowsPerWorkgroup, vectorized)
	if err != nil {
		return nil, err
	}
	l := &Linear{engine: e, format: format, rows: rows, cols: cols, workgroupSize: workgroupSize, rowsPerWorkgroup: rowsPerWorkgroup, vectorized: vectorized, kernel: kernel}
	defer func() {
		if err != nil {
			l.Close()
		}
	}()
	if uploadChunkBytes <= 0 {
		uploadChunkBytes = 16 << 20
	}
	for row0 := 0; row0 < rows; {
		count := min(maxRows, rows-row0)
		weightBytes := count * weightRowBytes
		chunk := linearChunk{row0: row0, rows: count}
		if chunk.weight, err = e.NewBuffer("gophonic-linear-weights", uint64(weightBytes), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
			return nil, err
		}
		l.chunks = append(l.chunks, chunk)
		if err = e.UploadBounded(chunk.weight, 0, weights[row0*weightRowBytes:(row0+count)*weightRowBytes], uploadChunkBytes); err != nil {
			return nil, err
		}
		if format == LinearQ8B {
			scaleBytes := count * scaleRowBytes
			paddedScaleBytes := (scaleBytes + 3) &^ 3
			if chunk.scale, err = e.NewBuffer("gophonic-linear-scales", uint64(paddedScaleBytes), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
				return nil, err
			}
			l.chunks[len(l.chunks)-1].scale = chunk.scale
			data := scales[row0*scaleRowBytes : (row0+count)*scaleRowBytes]
			if paddedScaleBytes != scaleBytes {
				padded := make([]byte, paddedScaleBytes)
				copy(padded, data)
				data = padded
			}
			if err = e.UploadBounded(chunk.scale, 0, data, uploadChunkBytes); err != nil {
				return nil, err
			}
		}
		row0 += count
	}
	return l, nil
}

// NewWorkspace allocates one lane's input, output, params, bind groups, and
// readback staging for this matrix. A workspace must not be used concurrently.
func (l *Linear) NewWorkspace() (*LinearWorkspace, error) {
	return l.newWorkspace(nil, nil)
}

func (l *Linear) newWorkspace(sharedInput, sharedOutput *wgpu.Buffer) (*LinearWorkspace, error) {
	if l == nil || l.closed || l.engine == nil || l.engine.device == nil {
		return nil, errors.New("gpuportable: linear is closed")
	}
	w := &LinearWorkspace{linear: l, inputBytes: make([]byte, l.cols*4), outputBytes: make([]byte, l.rows*4)}
	defer func() {
		if w != nil && w.linear == nil {
			w.Close()
		}
	}()
	var err error
	if sharedInput != nil {
		if sharedInput.Size() < uint64(l.cols*4) {
			w.linear = nil
			w.Close()
			return nil, errors.New("gpuportable: shared linear input buffer is too small")
		}
		w.input = sharedInput
	} else {
		if w.input, err = l.engine.NewBuffer("gophonic-linear-input", uint64(l.cols*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
			w.linear = nil
			w.Close()
			return nil, err
		}
		w.ownsInput = true
	}
	if sharedOutput != nil {
		if sharedOutput.Size() < uint64(l.rows*4) {
			w.linear = nil
			w.Close()
			return nil, errors.New("gpuportable: shared linear output buffer is too small")
		}
		w.output = sharedOutput
	} else {
		if w.output, err = l.engine.NewBuffer("gophonic-linear-output", uint64(l.rows*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc); err != nil {
			w.linear = nil
			w.Close()
			return nil, err
		}
		w.ownsOutput = true
	}
	if len(l.chunks) > maxIntForLinear()/linearParamStride {
		w.linear = nil
		w.Close()
		return nil, errors.New("gpuportable: too many projection chunks")
	}
	paramBytes := max(linearParamStride, len(l.chunks)*linearParamStride)
	if w.params, err = l.engine.NewBuffer("gophonic-linear-params", uint64(paramBytes), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		w.linear = nil
		w.Close()
		return nil, err
	}
	paramData := make([]byte, paramBytes)
	w.groups = make([]*wgpu.BindGroup, len(l.chunks))
	w.dispatches = make([]Dispatch, len(l.chunks))
	for i, chunk := range l.chunks {
		off := i * linearParamStride
		binary.LittleEndian.PutUint32(paramData[off:], uint32(chunk.rows))
		binary.LittleEndian.PutUint32(paramData[off+4:], uint32(l.cols))
		binary.LittleEndian.PutUint32(paramData[off+8:], uint32(chunk.row0))
		var group *wgpu.BindGroup
		if l.format == LinearBF16 {
			group, err = l.kernel.Bind("gophonic-linear-lane", BufferRange{Buffer: chunk.weight, Size: chunk.weight.Size()},
				BufferRange{Buffer: w.input, Size: w.input.Size()}, BufferRange{Buffer: w.output, Size: w.output.Size()},
				BufferRange{Buffer: w.params, Offset: uint64(off), Size: 16})
		} else {
			group, err = l.kernel.Bind("gophonic-linear-lane", BufferRange{Buffer: chunk.weight, Size: chunk.weight.Size()},
				BufferRange{Buffer: chunk.scale, Size: chunk.scale.Size()}, BufferRange{Buffer: w.input, Size: w.input.Size()},
				BufferRange{Buffer: w.output, Size: w.output.Size()}, BufferRange{Buffer: w.params, Offset: uint64(off), Size: 16})
		}
		if err != nil {
			w.linear = nil
			w.Close()
			return nil, err
		}
		w.groups[i] = group
		tilesPerWorkgroup := uint64(1)
		if l.vectorized {
			tilesPerWorkgroup = uint64(l.workgroupSize / linearLogicalTileSize)
		}
		rowsPerGroup := tilesPerWorkgroup * uint64(l.rowsPerWorkgroup)
		groups := (uint64(chunk.rows) + rowsPerGroup - 1) / rowsPerGroup
		w.dispatches[i] = Dispatch{Kernel: l.kernel, BindGroup: group, Workgroups: [3]uint32{uint32(groups), 1, 1}}
	}
	if err := l.engine.UploadBounded(w.params, 0, paramData, min(paramBytes, 16<<20)); err != nil {
		w.linear = nil
		w.Close()
		return nil, err
	}
	return w, nil
}

// UploadInput updates this lane's input vector without submitting work.
func (w *LinearWorkspace) UploadInput(input []float32) error {
	if w == nil || w.linear == nil || len(input) != w.linear.cols {
		return errors.New("gpuportable: invalid linear input")
	}
	for i, value := range input {
		binary.LittleEndian.PutUint32(w.inputBytes[i*4:], math.Float32bits(value))
	}
	return w.linear.engine.Upload(w.input, 0, w.inputBytes)
}

// Readback dispatches the projection and copies the complete output vector to
// dst in the same command submission. The final map is the only host wait.
func (w *LinearWorkspace) Readback(dst []float32) error {
	if w == nil || w.linear == nil || len(dst) != w.linear.rows {
		return errors.New("gpuportable: invalid linear output")
	}
	if err := w.linear.engine.SubmitReadback(w.dispatches, w.output, 0, w.outputBytes, &w.staging); err != nil {
		return err
	}
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(w.outputBytes[i*4:]))
	}
	return nil
}

// Run uploads one input vector, executes the projection, and reads its result.
func (w *LinearWorkspace) Run(input, output []float32) error {
	if err := w.UploadInput(input); err != nil {
		return err
	}
	return w.Readback(output)
}

// Close releases this lane's device allocations. Repeated calls are harmless.
func (w *LinearWorkspace) Close() {
	if w == nil {
		return
	}
	for _, group := range w.groups {
		if group != nil {
			group.Release()
		}
	}
	for _, b := range []*wgpu.Buffer{w.params, w.staging} {
		if b != nil {
			b.Release()
		}
	}
	if w.ownsInput && w.input != nil {
		w.input.Release()
	}
	if w.ownsOutput && w.output != nil {
		w.output.Release()
	}
	w.linear, w.groups, w.dispatches = nil, nil, nil
	w.input, w.output, w.params, w.staging = nil, nil, nil, nil
	w.ownsInput, w.ownsOutput = false, false
	w.inputBytes, w.outputBytes = nil, nil
}

// Close releases all immutable buffers owned by the matrix. Its workspaces
// must be closed first.
func (l *Linear) Close() {
	if l == nil || l.closed {
		return
	}
	for i := range l.chunks {
		if l.chunks[i].weight != nil {
			l.chunks[i].weight.Release()
		}
		if l.chunks[i].scale != nil {
			l.chunks[i].scale.Release()
		}
	}
	l.chunks, l.kernel, l.engine, l.closed = nil, nil, nil, true
}

type LinearWorkspace struct {
	linear      *Linear
	input       *wgpu.Buffer
	output      *wgpu.Buffer
	ownsInput   bool
	ownsOutput  bool
	params      *wgpu.Buffer
	staging     *wgpu.Buffer
	groups      []*wgpu.BindGroup
	dispatches  []Dispatch
	inputBytes  []byte
	outputBytes []byte
}

func maxIntForLinear() int { return int(^uint(0) >> 1) }
