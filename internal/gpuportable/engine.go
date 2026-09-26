// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Package gpuportable contains the small device layer shared by portable
// WGSL inference backends. It deliberately exposes buffers, kernels, and
// command recording instead of pretending that model operators are portable.
package gpuportable

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends"
)

var (
	ErrUnavailable       = errors.New("gpuportable: no compatible GPU adapter")
	ErrRustBindingBroken = errors.New("gpuportable: the wgpu rust-tag binding is unsupported by the pinned go-webgpu ABI")
)

// Engine owns one WebGPU instance, adapter, device, and queue. Models share
// an Engine and its immutable buffers; workspaces own their mutable buffers.
type Engine struct {
	instance      *wgpu.Instance
	adapter       *wgpu.Adapter
	device        *wgpu.Device
	info          gputypes.AdapterInfo
	limits        gputypes.Limits
	mu            sync.Mutex
	linearKernels map[linearKernelKey]*Kernel
	closed        bool
}

// New opens the default high-performance adapter. It requests no optional
// features, so its shaders can run on the pure-Go Metal, Vulkan, and DX12
// adapters without f16 or subgroup support.
func New() (*Engine, error) {
	if err := runtimeBuildError(); err != nil {
		return nil, err
	}
	i, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("gpuportable: create instance: %w", err)
	}
	a, err := i.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceHighPerformance})
	if err != nil {
		i.Release()
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	info := a.Info()
	if !acceleratedAdapter(info, os.Getenv("GOPHONIC_GPU_ALLOW_CPU_VULKAN") == "1") {
		a.Release()
		i.Release()
		return nil, fmt.Errorf("%w: adapter %q uses %s on a non-accelerated backend", ErrUnavailable, info.Name, info.Backend)
	}
	d, err := a.RequestDevice(nil)
	if err != nil {
		a.Release()
		i.Release()
		return nil, fmt.Errorf("gpuportable: request device: %w", err)
	}
	limits := d.Limits()
	if limits.MaxStorageBufferBindingSize == 0 || limits.MaxBufferSize == 0 {
		d.Release()
		a.Release()
		i.Release()
		return nil, errors.New("gpuportable: adapter did not report storage-buffer limits")
	}
	return &Engine{instance: i, adapter: a, device: d, info: info, limits: limits}, nil
}

func acceleratedAdapter(info gputypes.AdapterInfo, allowVulkanCPU bool) bool {
	if info.DeviceType == gputypes.DeviceTypeCPU {
		return allowVulkanCPU && info.Backend == gputypes.BackendVulkan
	}
	switch info.Backend {
	case gputypes.BackendMetal, gputypes.BackendVulkan, gputypes.BackendDX12:
		return true
	default:
		return false
	}
}

// Name reports the selected adapter.
func (e *Engine) Name() string {
	if e == nil {
		return ""
	}
	return e.info.Name
}

// AdapterInfo reports the selected GPU and graphics backend.
func (e *Engine) AdapterInfo() gputypes.AdapterInfo {
	if e == nil {
		return gputypes.AdapterInfo{}
	}
	return e.info
}

// Limits reports the selected adapter's actual limits.
func (e *Engine) Limits() gputypes.Limits {
	if e == nil {
		return gputypes.Limits{}
	}
	return e.limits
}

// Device returns the WebGPU device for model-specific resource creation.
func (e *Engine) Device() *wgpu.Device {
	if e == nil {
		return nil
	}
	return e.device
}

// Close waits for queued work and releases the adapter.
func (e *Engine) Close() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return
	}
	e.closed = true
	_ = e.device.WaitIdle()
	for key, k := range e.linearKernels {
		if k != nil {
			k.Close()
			delete(e.linearKernels, key)
		}
	}
	e.device.Release()
	e.adapter.Release()
	e.instance.Release()
	e.device, e.adapter, e.instance = nil, nil, nil
}

// NewBuffer allocates a device buffer with the requested usages.
func (e *Engine) NewBuffer(label string, size uint64, usage gputypes.BufferUsage) (*wgpu.Buffer, error) {
	if e == nil || e.device == nil || size == 0 || size > e.limits.MaxBufferSize {
		return nil, fmt.Errorf("gpuportable: buffer size %d is unavailable", size)
	}
	b, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	if err != nil {
		return nil, fmt.Errorf("gpuportable: create buffer %q (%d bytes): %w", label, size, err)
	}
	return b, nil
}

// Upload copies aligned bytes into a persistent GPU buffer.
func (e *Engine) Upload(dst *wgpu.Buffer, offset uint64, src []byte) error {
	if e == nil || e.device == nil || dst == nil {
		return errors.New("gpuportable: upload to a nil buffer")
	}
	size := uint64(len(src))
	if len(src)%4 != 0 || offset%4 != 0 || offset > dst.Size() || size > dst.Size()-offset {
		return fmt.Errorf("gpuportable: invalid upload range at %d with %d bytes", offset, len(src))
	}
	if len(src) == 0 {
		return nil
	}
	if err := e.device.Queue().WriteBuffer(dst, offset, src); err != nil {
		return fmt.Errorf("gpuportable: upload: %w", err)
	}
	return nil
}

// UploadBounded uploads large immutable payloads in bounded submissions. It
// waits for each chunk's copy before reusing driver staging, so a model loader
// never queues a second model-sized staging allocation.
func (e *Engine) UploadBounded(dst *wgpu.Buffer, offset uint64, src []byte, maxChunkBytes int) error {
	if e == nil || e.device == nil || dst == nil {
		return errors.New("gpuportable: upload to a nil buffer")
	}
	if offset%4 != 0 || offset > dst.Size() || uint64(len(src)) > dst.Size()-offset {
		return fmt.Errorf("gpuportable: invalid bounded upload range at %d with %d bytes", offset, len(src))
	}
	if len(src)%4 != 0 {
		return fmt.Errorf("gpuportable: bounded upload size %d is not four-byte aligned", len(src))
	}
	if maxChunkBytes <= 0 {
		maxChunkBytes = 16 << 20
	}
	chunk := uint64(maxChunkBytes &^ 3)
	if chunk == 0 {
		return errors.New("gpuportable: bounded upload chunk must be at least four bytes")
	}
	for done := uint64(0); done < uint64(len(src)); {
		n := min(chunk, uint64(len(src))-done)
		if err := e.Upload(dst, offset+done, src[int(done):int(done+n)]); err != nil {
			return err
		}
		if err := e.Submit(); err != nil {
			return err
		}
		if err := e.device.WaitIdle(); err != nil {
			return fmt.Errorf("gpuportable: wait for upload chunk: %w", err)
		}
		done += n
	}
	return nil
}

// Binding describes one kernel buffer binding.
type Binding uint8

const (
	BindingReadOnlyStorage Binding = iota + 1
	BindingStorage
	BindingUniform
)

// Kernel owns a WGSL module, bind-group layout, pipeline layout, and compute
// pipeline. Bind groups are built separately and can be retained per lane.
type Kernel struct {
	device                *wgpu.Device
	module                *wgpu.ShaderModule
	bind                  *wgpu.BindGroupLayout
	layout                *wgpu.PipelineLayout
	pipeline              *wgpu.ComputePipeline
	bindings              []Binding
	maxStorageBindingSize uint64
}

// NewKernel compiles source for one entry point and its ordered buffer bindings.
func (e *Engine) NewKernel(label, source, entry string, bindings []Binding) (*Kernel, error) {
	if e == nil || e.device == nil {
		return nil, errors.New("gpuportable: nil engine")
	}
	k := &Kernel{device: e.device, bindings: append([]Binding(nil), bindings...), maxStorageBindingSize: e.limits.MaxStorageBufferBindingSize}
	fail := func(err error) (*Kernel, error) {
		k.Close()
		return nil, err
	}
	var err error
	if k.module, err = e.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: label, WGSL: source}); err != nil {
		return fail(fmt.Errorf("gpuportable: compile %s: %w", label, err))
	}
	entries := make([]gputypes.BindGroupLayoutEntry, len(bindings))
	for i, binding := range bindings {
		var typ gputypes.BufferBindingType
		switch binding {
		case BindingReadOnlyStorage:
			typ = gputypes.BufferBindingTypeReadOnlyStorage
		case BindingStorage:
			typ = gputypes.BufferBindingTypeStorage
		case BindingUniform:
			typ = gputypes.BufferBindingTypeUniform
		default:
			return fail(fmt.Errorf("gpuportable: invalid binding type %d", binding))
		}
		entries[i] = gputypes.BindGroupLayoutEntry{
			Binding: uint32(i), Visibility: wgpu.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: typ},
		}
	}
	if k.bind, err = e.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Label: label + "-bindings", Entries: entries}); err != nil {
		return fail(fmt.Errorf("gpuportable: layout %s: %w", label, err))
	}
	if k.layout, err = e.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label: label + "-pipeline-layout", BindGroupLayouts: []*wgpu.BindGroupLayout{k.bind},
	}); err != nil {
		return fail(fmt.Errorf("gpuportable: pipeline layout %s: %w", label, err))
	}
	if k.pipeline, err = e.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: label, Layout: k.layout, Module: k.module, EntryPoint: entry,
	}); err != nil {
		return fail(fmt.Errorf("gpuportable: pipeline %s: %w", label, err))
	}
	return k, nil
}

func (e *Engine) linearKernel(format LinearFormat, workgroupSize, rowsPerWorkgroup uint32, vectorized bool) (*Kernel, error) {
	if e == nil || e.device == nil || format == 0 {
		return nil, errors.New("gpuportable: invalid linear format or nil engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("gpuportable: engine is closed")
	}
	if workgroupSize < 32 || workgroupSize > e.limits.MaxComputeWorkgroupSizeX || workgroupSize > e.limits.MaxComputeInvocationsPerWorkgroup || workgroupSize&(workgroupSize-1) != 0 ||
		rowsPerWorkgroup != 1 && rowsPerWorkgroup != 2 && rowsPerWorkgroup != 4 {
		return nil, fmt.Errorf("gpuportable: unsupported linear geometry workgroup=%d rows=%d", workgroupSize, rowsPerWorkgroup)
	}
	key := linearKernelKey{format: format, workgroupSize: workgroupSize, rowsPerWorkgroup: rowsPerWorkgroup, vectorized: vectorized}
	if k := e.linearKernels[key]; k != nil {
		return k, nil
	}
	bindings := []Binding{BindingReadOnlyStorage, BindingReadOnlyStorage, BindingStorage, BindingReadOnlyStorage}
	if format == LinearQ8B {
		bindings = []Binding{BindingReadOnlyStorage, BindingReadOnlyStorage, BindingReadOnlyStorage, BindingStorage, BindingReadOnlyStorage}
	}
	label := "gophonic-bf16-gemv"
	if format == LinearQ8B {
		label = "gophonic-q8b-gemv"
	}
	label = fmt.Sprintf("%s-wg%d-r%d", label, workgroupSize, rowsPerWorkgroup)
	k, err := e.NewKernel(label, linearWGSL(format, workgroupSize, rowsPerWorkgroup, vectorized), "gemv", bindings)
	if err != nil {
		return nil, err
	}
	if e.linearKernels == nil {
		e.linearKernels = make(map[linearKernelKey]*Kernel)
	}
	e.linearKernels[key] = k
	return k, nil
}

// Bind creates a bind group over the supplied buffers.
func (k *Kernel) Bind(label string, buffers ...BufferRange) (*wgpu.BindGroup, error) {
	if k == nil || k.device == nil || len(buffers) == 0 {
		return nil, errors.New("gpuportable: invalid kernel bindings")
	}
	entries := make([]wgpu.BindGroupEntry, len(buffers))
	for i, b := range buffers {
		if b.Buffer == nil {
			return nil, fmt.Errorf("gpuportable: nil buffer at binding %d", i)
		}
		if i >= len(k.bindings) || b.Offset > b.Buffer.Size() {
			return nil, fmt.Errorf("gpuportable: invalid range at binding %d", i)
		}
		size := b.Size
		if size == 0 {
			size = b.Buffer.Size() - b.Offset
		}
		if size == 0 || size > b.Buffer.Size()-b.Offset {
			return nil, fmt.Errorf("gpuportable: invalid range at binding %d", i)
		}
		if (k.bindings[i] == BindingReadOnlyStorage || k.bindings[i] == BindingStorage) && size > k.maxStorageBindingSize {
			return nil, fmt.Errorf("gpuportable: binding %d size %d exceeds adapter storage binding limit %d", i, size, k.maxStorageBindingSize)
		}
		entries[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b.Buffer, Offset: b.Offset, Size: size}
	}
	if len(buffers) != len(k.bindings) {
		return nil, fmt.Errorf("gpuportable: kernel expects %d bindings, received %d", len(k.bindings), len(buffers))
	}
	group, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Label: label, Layout: k.bind, Entries: entries})
	if err != nil {
		return nil, fmt.Errorf("gpuportable: bind %s: %w", label, err)
	}
	return group, nil
}

// BufferRange binds a byte range of a GPU buffer. A zero size means the
// remaining buffer from Offset.
type BufferRange struct {
	Buffer *wgpu.Buffer
	Offset uint64
	Size   uint64
}

// Dispatch is one compute pass. Separate passes provide a global memory
// ordering point on the pinned pure-Go Vulkan backend.
type Dispatch struct {
	Kernel     *Kernel
	BindGroup  *wgpu.BindGroup
	Workgroups [3]uint32
}

// Submit records each dispatch as its own compute pass and submits one command
// buffer. This preserves dependencies without waiting on the CPU between ops.
func (e *Engine) Submit(dispatches ...Dispatch) error {
	return e.submit(dispatches, nil, false)
}

// SubmitIndependent records dispatches with no inter-dispatch data dependency
// in one compute pass. Callers must ensure their writes do not overlap.
func (e *Engine) SubmitIndependent(dispatches ...Dispatch) error {
	return e.submit(dispatches, nil, true)
}

// SubmitReadback records the dispatches and output copy into one queue
// submission, then waits for only the requested output to become readable.
func (e *Engine) SubmitReadback(dispatches []Dispatch, src *wgpu.Buffer, offset uint64, dst []byte, staging **wgpu.Buffer) error {
	if e == nil || e.device == nil || src == nil || staging == nil || len(dst) == 0 {
		return errors.New("gpuportable: invalid readback")
	}
	size := (uint64(len(dst)) + 3) &^ uint64(3)
	if offset%4 != 0 || offset > src.Size() || size > src.Size()-offset {
		return fmt.Errorf("gpuportable: invalid readback range at %d with %d bytes", offset, len(dst))
	}
	if *staging == nil || (*staging).Size() < size {
		b, err := e.NewBuffer("gophonic-readback", size, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
		if err != nil {
			return err
		}
		old := *staging
		*staging = b
		if old != nil {
			old.Release()
		}
	}
	copyOp := &bufferCopy{src: src, srcOffset: offset, dst: *staging, size: size}
	if err := e.submit(dispatches, copyOp, false); err != nil {
		return err
	}
	return e.mapReadback(*staging, size, dst)
}

// SubmitIndependentReadback batches independent dispatches into one compute
// pass and copies one result in the same queue submission. The caller must
// ensure dispatches have no dependencies and their writes do not overlap.
func (e *Engine) SubmitIndependentReadback(dispatches []Dispatch, src *wgpu.Buffer, offset uint64, dst []byte, staging **wgpu.Buffer) error {
	if e == nil || e.device == nil || src == nil || staging == nil || len(dst) == 0 {
		return errors.New("gpuportable: invalid readback")
	}
	size := (uint64(len(dst)) + 3) &^ uint64(3)
	if offset%4 != 0 || offset > src.Size() || size > src.Size()-offset {
		return fmt.Errorf("gpuportable: invalid readback range at %d with %d bytes", offset, len(dst))
	}
	if *staging == nil || (*staging).Size() < size {
		b, err := e.NewBuffer("gophonic-readback", size, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
		if err != nil {
			return err
		}
		old := *staging
		*staging = b
		if old != nil {
			old.Release()
		}
	}
	copyOp := &bufferCopy{src: src, srcOffset: offset, dst: *staging, size: size}
	if err := e.submit(dispatches, copyOp, true); err != nil {
		return err
	}
	return e.mapReadback(*staging, size, dst)
}

func (e *Engine) mapReadback(staging *wgpu.Buffer, size uint64, dst []byte) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := staging.Map(ctx, wgpu.MapModeRead, 0, size); err != nil {
		return fmt.Errorf("gpuportable: map readback: %w", err)
	}
	defer staging.Unmap()
	mapped, err := staging.MappedRange(0, size)
	if err != nil {
		return fmt.Errorf("gpuportable: read mapped result: %w", err)
	}
	defer mapped.Release()
	copy(dst, mapped.Bytes()[:len(dst)])
	return nil
}

// Readback copies a result into host memory with one copy submission.
func (e *Engine) Readback(src *wgpu.Buffer, offset uint64, dst []byte, staging **wgpu.Buffer) error {
	return e.SubmitReadback(nil, src, offset, dst, staging)
}

type bufferCopy struct {
	src       *wgpu.Buffer
	srcOffset uint64
	dst       *wgpu.Buffer
	size      uint64
}

func (e *Engine) submit(dispatches []Dispatch, copyOp *bufferCopy, independent bool) error {
	if e == nil || e.device == nil {
		return errors.New("gpuportable: nil engine")
	}
	for i, d := range dispatches {
		if d.Kernel == nil || d.Kernel.device != e.device || d.Kernel.pipeline == nil || d.BindGroup == nil || d.Workgroups[0] == 0 || d.Workgroups[1] == 0 || d.Workgroups[2] == 0 {
			return fmt.Errorf("gpuportable: invalid dispatch %d", i)
		}
	}
	encoder, err := e.device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "gophonic"})
	if err != nil {
		return fmt.Errorf("gpuportable: create command encoder: %w", err)
	}
	finished := false
	defer func() {
		if !finished {
			encoder.DiscardEncoding()
		}
	}()
	if independent && len(dispatches) != 0 {
		pass, err := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{Label: "gophonic-independent-dispatches"})
		if err != nil {
			return fmt.Errorf("gpuportable: begin independent compute pass: %w", err)
		}
		var currentPipeline *wgpu.ComputePipeline
		for _, d := range dispatches {
			if d.Kernel.pipeline != currentPipeline {
				pass.SetPipeline(d.Kernel.pipeline)
				currentPipeline = d.Kernel.pipeline
			}
			pass.SetBindGroup(0, d.BindGroup, nil)
			pass.Dispatch(d.Workgroups[0], d.Workgroups[1], d.Workgroups[2])
		}
		if err := pass.End(); err != nil {
			return fmt.Errorf("gpuportable: end independent compute pass: %w", err)
		}
	} else {
		for i, d := range dispatches {
			pass, err := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{Label: "gophonic-dispatch"})
			if err != nil {
				return fmt.Errorf("gpuportable: begin compute pass %d: %w", i, err)
			}
			pass.SetPipeline(d.Kernel.pipeline)
			pass.SetBindGroup(0, d.BindGroup, nil)
			pass.Dispatch(d.Workgroups[0], d.Workgroups[1], d.Workgroups[2])
			if err := pass.End(); err != nil {
				return fmt.Errorf("gpuportable: end compute pass %d: %w", i, err)
			}
		}
	}
	if copyOp != nil {
		encoder.CopyBufferToBuffer(copyOp.src, copyOp.srcOffset, copyOp.dst, 0, copyOp.size)
	}
	command, err := encoder.Finish()
	finished = true
	if err != nil {
		return fmt.Errorf("gpuportable: finish command buffer: %w", err)
	}
	defer command.Release()
	if _, err := e.device.Queue().Submit(command); err != nil {
		return fmt.Errorf("gpuportable: submit: %w", err)
	}
	return nil
}

// Close releases the handles owned by a Kernel.
func (k *Kernel) Close() {
	if k == nil {
		return
	}
	if k.pipeline != nil {
		k.pipeline.Release()
	}
	if k.layout != nil {
		k.layout.Release()
	}
	if k.bind != nil {
		k.bind.Release()
	}
	if k.module != nil {
		k.module.Release()
	}
	k.pipeline, k.layout, k.bind, k.module = nil, nil, nil, nil
	k.device = nil
}
