// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

// Package metal is a minimal Metal compute binding in pure Go. It loads
// Metal and the Objective-C runtime with dlopen and calls objc_msgSend
// through the Go runtime's libc trampoline, so it needs no cgo. Only what a
// compute pipeline needs is bound: shared buffers, pipelines compiled from
// source, and serial or concurrent compute encoders. Encoding a dispatch
// does not allocate.
package metal

import (
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

const (
	rtldNow    = 0x2
	rtldGlobal = 0x8
)

var (
	loadOnce sync.Once
	loadErr  error

	msgSend, poolPush, poolPop, createDevice uintptr

	clsNSString uintptr

	selName, selUTF8, selStringWithUTF8, selRelease, selRetain,
	selNewQueue, selNewLibrary, selNewFunction, selNewPipeline, selNewBuffer,
	selContents, selLength, selMaxThreads, selExecWidth, selDescription,
	selCommandBuffer, selEncoder, selEncoderType, selCommit, selWait, selStatus, selError,
	selSetPipeline, selSetBuffer, selSetBytes, selDispatch, selDispatchThreads, selBarrier, selEndEncoding,
	selSetThreadgroupMemory uintptr
)

func cstring(s string) *byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0]
}

func dlopen(path string) uintptr {
	h, _, _ := call6(libc_dlopen_trampoline_addr, uintptr(unsafe.Pointer(cstring(path))), rtldNow|rtldGlobal, 0, 0, 0, 0)
	return h
}

func dlsym(h uintptr, name string) uintptr {
	p, _, _ := call6(libc_dlsym_trampoline_addr, h, uintptr(unsafe.Pointer(cstring(name))), 0, 0, 0, 0)
	return p
}

func load() error {
	metal := dlopen("/System/Library/Frameworks/Metal.framework/Metal")
	objc := dlopen("/usr/lib/libobjc.A.dylib")
	foundation := dlopen("/System/Library/Frameworks/Foundation.framework/Foundation")
	if metal == 0 || objc == 0 || foundation == 0 {
		return errors.New("metal: cannot load Metal, Foundation, or libobjc")
	}
	msgSend = dlsym(objc, "objc_msgSend")
	poolPush = dlsym(objc, "objc_autoreleasePoolPush")
	poolPop = dlsym(objc, "objc_autoreleasePoolPop")
	createDevice = dlsym(metal, "MTLCreateSystemDefaultDevice")
	getClass := dlsym(objc, "objc_getClass")
	register := dlsym(objc, "sel_registerName")
	if msgSend == 0 || poolPush == 0 || poolPop == 0 || createDevice == 0 || getClass == 0 || register == 0 {
		return errors.New("metal: missing runtime symbols")
	}
	clsNSString, _, _ = call6(getClass, uintptr(unsafe.Pointer(cstring("NSString"))), 0, 0, 0, 0, 0)
	sel := func(name string) uintptr {
		s, _, _ := call6(register, uintptr(unsafe.Pointer(cstring(name))), 0, 0, 0, 0, 0)
		return s
	}
	selName = sel("name")
	selUTF8 = sel("UTF8String")
	selStringWithUTF8 = sel("stringWithUTF8String:")
	selRelease = sel("release")
	selRetain = sel("retain")
	selNewQueue = sel("newCommandQueue")
	selNewLibrary = sel("newLibraryWithSource:options:error:")
	selNewFunction = sel("newFunctionWithName:")
	selNewPipeline = sel("newComputePipelineStateWithFunction:error:")
	selNewBuffer = sel("newBufferWithLength:options:")
	selContents = sel("contents")
	selLength = sel("length")
	selMaxThreads = sel("maxTotalThreadsPerThreadgroup")
	selExecWidth = sel("threadExecutionWidth")
	selDescription = sel("localizedDescription")
	selCommandBuffer = sel("commandBufferWithUnretainedReferences")
	selEncoder = sel("computeCommandEncoder")
	selEncoderType = sel("computeCommandEncoderWithDispatchType:")
	selCommit = sel("commit")
	selWait = sel("waitUntilCompleted")
	selStatus = sel("status")
	selError = sel("error")
	selSetPipeline = sel("setComputePipelineState:")
	selSetBuffer = sel("setBuffer:offset:atIndex:")
	selSetBytes = sel("setBytes:length:atIndex:")
	selDispatch = sel("dispatchThreadgroups:threadsPerThreadgroup:")
	selDispatchThreads = sel("dispatchThreads:threadsPerThreadgroup:")
	selBarrier = sel("memoryBarrierWithScope:")
	selEndEncoding = sel("endEncoding")
	selSetThreadgroupMemory = sel("setThreadgroupMemoryLength:atIndex:")
	if clsNSString == 0 {
		return errors.New("metal: missing NSString")
	}
	return nil
}

func send(obj, sel uintptr, args ...uintptr) uintptr {
	var a [4]uintptr
	copy(a[:], args)
	r, _, _ := call6(msgSend, obj, sel, a[0], a[1], a[2], a[3])
	return r
}

func send0(obj, sel uintptr) uintptr {
	r, _, _ := call6(msgSend, obj, sel, 0, 0, 0, 0)
	return r
}

// pointer converts a C address, which the Go heap never owns, to a pointer.
func pointer(p uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&p)) }

func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	s := pointer(p)
	n := 0
	for *(*byte)(unsafe.Add(s, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(s), n))
}

func nsString(s string) uintptr {
	c := cstring(s)
	r := send(clsNSString, selStringWithUTF8, uintptr(unsafe.Pointer(c)))
	runtime.KeepAlive(c)
	return r
}

func nsError(err uintptr) error {
	if err == 0 {
		return errors.New("metal: unknown error")
	}
	return errors.New("metal: " + goString(send0(send0(err, selDescription), selUTF8)))
}

// withPool runs f inside an autorelease pool on a locked OS thread.
func withPool(f func()) {
	runtime.LockOSThread()
	p, _, _ := call6(poolPush, 0, 0, 0, 0, 0, 0)
	f()
	call6(poolPop, p, 0, 0, 0, 0, 0)
	runtime.UnlockOSThread()
}

// Device is the system default GPU with one command queue.
type Device struct {
	dev, queue uintptr
	name       string
}

// Open returns the system default Metal device.
func Open() (*Device, error) {
	loadOnce.Do(func() { loadErr = load() })
	if loadErr != nil {
		return nil, loadErr
	}
	d := &Device{}
	d.dev, _, _ = call6(createDevice, 0, 0, 0, 0, 0, 0)
	if d.dev == 0 {
		return nil, errors.New("metal: no device")
	}
	withPool(func() {
		d.name = goString(send0(send0(d.dev, selName), selUTF8))
		d.queue = send0(d.dev, selNewQueue)
	})
	if d.queue == 0 {
		return nil, errors.New("metal: cannot create a command queue")
	}
	return d, nil
}

// Name reports the device name.
func (d *Device) Name() string { return d.name }

// Close releases the queue and device.
func (d *Device) Close() {
	if d == nil || d.dev == 0 {
		return
	}
	send0(d.queue, selRelease)
	send0(d.dev, selRelease)
	d.dev, d.queue = 0, 0
}

// Library is compiled Metal shading language source.
type Library struct{ lib uintptr }

// Compile compiles Metal shading language source.
func (d *Device) Compile(src string) (*Library, error) {
	var lib uintptr
	var err error
	withPool(func() {
		var e uintptr
		lib = send(d.dev, selNewLibrary, nsString(src), 0, uintptr(unsafe.Pointer(&e)))
		if lib == 0 {
			err = nsError(e)
		}
	})
	if err != nil {
		return nil, err
	}
	return &Library{lib: lib}, nil
}

// Pipeline is a compute pipeline for one kernel function.
type Pipeline struct {
	p                     uintptr
	MaxThreads, SIMDWidth int
}

// Pipeline builds a compute pipeline for the named kernel.
func (d *Device) Pipeline(l *Library, name string) (*Pipeline, error) {
	var p *Pipeline
	var err error
	withPool(func() {
		fn := send(l.lib, selNewFunction, nsString(name))
		if fn == 0 {
			err = errors.New("metal: no kernel " + name)
			return
		}
		defer send0(fn, selRelease)
		var e uintptr
		ps := send(d.dev, selNewPipeline, fn, uintptr(unsafe.Pointer(&e)))
		if ps == 0 {
			err = nsError(e)
			return
		}
		p = &Pipeline{p: ps, MaxThreads: int(send0(ps, selMaxThreads)), SIMDWidth: int(send0(ps, selExecWidth))}
	})
	return p, err
}

// Buffer is a GPU buffer in shared (unified) memory.
type Buffer struct {
	b    uintptr
	data []byte
}

// Buffer allocates a zeroed shared buffer of n bytes.
func (d *Device) Buffer(n int) (*Buffer, error) {
	if n <= 0 {
		n = 16
	}
	b := send(d.dev, selNewBuffer, uintptr(n), 0) // MTLResourceStorageModeShared
	if b == 0 {
		return nil, errors.New("metal: cannot allocate buffer")
	}
	p := send0(b, selContents)
	return &Buffer{b: b, data: unsafe.Slice((*byte)(pointer(p)), n)}, nil
}

// Bytes returns the buffer's memory, shared with the GPU.
func (b *Buffer) Bytes() []byte { return b.data }

// Release frees the buffer.
func (b *Buffer) Release() {
	if b != nil && b.b != 0 {
		send0(b.b, selRelease)
		b.b, b.data = 0, nil
	}
}

// Size is a Metal grid or threadgroup size.
type Size struct{ X, Y, Z int }

// Encoder records compute dispatches into one command buffer. Use it from a
// single goroutine between Begin and Wait.
type Encoder struct {
	d          *Device
	cmd, enc   uintptr
	pool       uintptr
	groups, tg [3]uintptr
}

// Begin starts a command buffer with one compute encoder. Concurrent
// encoders may run dispatches out of order; separate dependent dispatches
// with Barrier.
func (d *Device) Begin(e *Encoder, concurrent bool) {
	runtime.LockOSThread()
	e.d = d
	e.pool, _, _ = call6(poolPush, 0, 0, 0, 0, 0, 0)
	e.cmd = send0(d.queue, selCommandBuffer)
	if concurrent {
		e.enc = send(e.cmd, selEncoderType, 1)
	} else {
		e.enc = send0(e.cmd, selEncoder)
	}
}

// SetPipeline selects the kernel for following dispatches.
func (e *Encoder) SetPipeline(p *Pipeline) {
	rawCall6(msgSend, e.enc, selSetPipeline, p.p, 0, 0, 0)
}

// SetBuffer binds b at byte offset to argument index.
func (e *Encoder) SetBuffer(b *Buffer, offset, index int) {
	rawCall6(msgSend, e.enc, selSetBuffer, b.b, uintptr(offset), uintptr(index), 0)
}

// SetBytes copies n bytes at p into argument index (at most 4 KiB).
func (e *Encoder) SetBytes(p unsafe.Pointer, n, index int) {
	rawCall6(msgSend, e.enc, selSetBytes, uintptr(p), uintptr(n), uintptr(index), 0)
}

// SetThreadgroupMemory reserves n bytes of threadgroup memory at index.
func (e *Encoder) SetThreadgroupMemory(n, index int) {
	rawCall6(msgSend, e.enc, selSetThreadgroupMemory, uintptr(n), uintptr(index), 0, 0)
}

// Dispatch runs groups threadgroups of threads each.
func (e *Encoder) Dispatch(groups, threads Size) {
	e.groups = [3]uintptr{uintptr(groups.X), uintptr(groups.Y), uintptr(groups.Z)}
	e.tg = [3]uintptr{uintptr(threads.X), uintptr(threads.Y), uintptr(threads.Z)}
	rawCall6(msgSend, e.enc, selDispatch, uintptr(unsafe.Pointer(&e.groups)), uintptr(unsafe.Pointer(&e.tg)), 0, 0)
}

// Barrier orders buffer accesses between dispatches of a concurrent encoder.
func (e *Encoder) Barrier() {
	rawCall6(msgSend, e.enc, selBarrier, 1, 0, 0, 0) // MTLBarrierScopeBuffers
}

// Wait ends encoding, commits, and waits for the GPU.
func (e *Encoder) Wait() error {
	send0(e.enc, selEndEncoding)
	send0(e.cmd, selCommit)
	send0(e.cmd, selWait)
	var err error
	if send0(e.cmd, selStatus) == 5 { // MTLCommandBufferStatusError
		err = nsError(send0(e.cmd, selError))
	}
	call6(poolPop, e.pool, 0, 0, 0, 0, 0)
	runtime.UnlockOSThread()
	e.cmd, e.enc, e.pool = 0, 0, 0
	return err
}
