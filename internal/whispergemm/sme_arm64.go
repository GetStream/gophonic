// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64

package whispergemm

import "sync/atomic"

// smeBlockRows is the number of A rows one SME call covers: two 16-lane
// streaming vectors, matching four 16x16 FP32 ZA accumulator tiles.
const smeBlockRows = 32

// smeMulBlock multiplies up to 32 rows of A by every packed panel of B in
// streaming mode. It transposes A through ZA into scratch, then accumulates
// each output in increasing K order with fused FP32 outer products. It
// returns how many tiles were recomputed after a signal cleared the upper
// lanes of the Z registers.
//
//go:noescape
func smeMulBlock(a *float32, lda, rows, k int, b *float32, n int, c *float32, ldc int, scratch *float32) (retries int)

// smeVectorBytes executes RDSVL. Call it only after smeSupported reports SME.
func smeVectorBytes() int

// smeEnabled requires SME with 512-bit streaming vectors, which the kernel's
// fixed 16-lane panels assume. Detection is automatic; tests may switch it
// off to compare the NEON or scalar kernels.
var smeEnabled = smeSupported() && smeVectorBytes() == 64

// smeRetries counts signal-induced tile recomputations for tests.
var smeRetries atomic.Int64

// smeScratch is a free list of packed-A buffers. Unlike sync.Pool it never
// drops warm buffers, so steady-state calls allocate nothing.
var smeScratch = make(chan *[]float32, 64)

// mulSME reports false when SME is unavailable and the caller must use the
// NEON or scalar kernels.
func mulSME(dst []float32, dstStride int, a []float32, aStride int, packed []float32, m, k, n int) bool {
	if !smeEnabled {
		return false
	}
	var buffer *[]float32
	select {
	case buffer = <-smeScratch:
	default:
		buffer = new([]float32)
	}
	need := (k + 15) / 16 * 16 * smeBlockRows
	if cap(*buffer) < need {
		*buffer = make([]float32, need)
	}
	scratch := (*buffer)[:need]
	retries := 0
	for r := 0; r < m; r += smeBlockRows {
		rows := min(smeBlockRows, m-r)
		retries += smeMulBlock(&a[r*aStride], aStride*4, rows, k, &packed[0], n, &dst[r*dstStride], dstStride*4, &scratch[0])
	}
	select {
	case smeScratch <- buffer:
	default:
	}
	if retries != 0 {
		smeRetries.Add(int64(retries))
	}
	return true
}

//go:noescape
func smeVectorF32(x *float32, k int, w *float32, n int, y *float32, kp int) (retries int)

//go:noescape
func smeVectorF16(x *float32, k int, w *uint16, n int, y *float32, kp int) (retries int)

func mulPackedVectorSME(p *PackedVector, dst, x []float32) bool {
	if !smeEnabled {
		return false
	}
	var retries int
	if p.w16 != nil {
		retries = smeVectorF16(&x[0], p.k, &p.w16[0], p.n, &dst[0], p.kp)
	} else {
		retries = smeVectorF32(&x[0], p.k, &p.w32[0], p.n, &dst[0], p.kp)
	}
	if retries != 0 {
		smeRetries.Add(int64(retries))
	}
	return true
}

// smeTransposePack writes PackedB panels from an N-by-K source through ZA.
//
//go:noescape
func smeTransposePack(src *float32, strideBytes, n, k int, dst *float32)
