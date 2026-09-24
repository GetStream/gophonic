// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
	"time"
	"unsafe"
)

const gemvSrc = `
#include <metal_stdlib>
using namespace metal;
// y[n] = scale[n] * dot(W[n,:], x). 8 simdgroups per threadgroup, R rows per simdgroup.
constant constexpr uint R = 2;
kernel void gemv_i8(device const char4 *W [[buffer(0)]], device const float *scale [[buffer(1)]],
	device const float4 *x [[buffer(2)]], device float *y [[buffer(3)]], constant uint &K [[buffer(4)]],
	uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
	uint row0 = (tg * 8 + sg) * R;
	uint k4 = K / 4;
	float acc[R] = {0};
	for (uint i = lane * 4; i < k4; i += 128) {
		float4 x0 = x[i], x1 = x[i+1], x2 = x[i+2], x3 = x[i+3];
		for (uint r = 0; r < R; r++) {
			device const char4 *w = W + (row0 + r) * k4 + i;
			acc[r] += dot(float4(w[0]), x0) + dot(float4(w[1]), x1) + dot(float4(w[2]), x2) + dot(float4(w[3]), x3);
		}
	}
	for (uint r = 0; r < R; r++) {
		float s = simd_sum(acc[r]);
		if (lane == 0) y[row0 + r] = s * scale[row0 + r];
	}
}
`

// TestGEMV checks a small int8 GEMV end to end through the binding.
func TestGEMV(t *testing.T) { gemvCheck(t, 2, 1) }

// BenchmarkGEMVBandwidth streams 1.6 GB of int8 weights per iteration.
func BenchmarkGEMVBandwidth(b *testing.B) { gemvCheck(b, 96, b.N) }

func gemvCheck(t testing.TB, mats, iters int) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	lib, err := d.Compile(gemvSrc)
	if err != nil {
		t.Fatal(err)
	}
	p, err := d.Pipeline(lib, "gemv_i8")
	if err != nil {
		t.Fatal(err)
	}
	const K, N = 4096, 4096
	w, _ := d.Buffer(K * N * mats)
	sc, _ := d.Buffer(4 * N * mats)
	x, _ := d.Buffer(4 * K)
	y, _ := d.Buffer(4 * N * mats)
	wb := w.Bytes()
	for i := range wb {
		wb[i] = byte(i * 7 % 13)
	}
	scales := unsafe.Slice((*float32)(unsafe.Pointer(&sc.Bytes()[0])), N*mats)
	for i := range scales {
		scales[i] = 1
	}
	xs := unsafe.Slice((*float32)(unsafe.Pointer(&x.Bytes()[0])), K)
	for i := range xs {
		xs[i] = float32(i%5) - 2
	}
	k := uint32(K)
	var e Encoder
	run := func() time.Duration {
		start := time.Now()
		d.Begin(&e, false)
		e.SetPipeline(p)
		e.SetBuffer(x, 0, 2)
		e.SetBytes(unsafe.Pointer(&k), 4, 4)
		for m := range mats {
			e.SetBuffer(w, m*K*N, 0)
			e.SetBuffer(sc, 4*m*N, 1)
			e.SetBuffer(y, 4*m*N, 3)
			e.Dispatch(Size{N / 16, 1, 1}, Size{256, 1, 1})
		}
		if err := e.Wait(); err != nil {
			t.Fatal(err)
		}
		return time.Since(start)
	}
	var best time.Duration = time.Hour
	for range iters {
		best = min(best, run())
	}
	ys := unsafe.Slice((*float32)(unsafe.Pointer(&y.Bytes()[0])), N*mats)
	for _, n := range []int{0, 1, 777, N*mats - 1} {
		var want float64
		for i := range K {
			want += float64(int8(wb[n*K+i])) * float64(xs[i])
		}
		if math.Abs(want-float64(ys[n])) > 1e-3*math.Abs(want)+1e-2 {
			t.Errorf("y[%d] = %v, want %v", n, ys[n], want)
		}
	}
	if b, ok := t.(*testing.B); ok {
		b.ReportMetric(float64(K*N*mats)/best.Seconds()/1e9, "GB/s")
	}
}
