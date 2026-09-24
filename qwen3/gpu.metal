// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Qwen3 forward kernels for Apple GPUs. The residual stream lives in a
// Hadamard-rotated basis, and RMSNorm weights and the per-head value
// rotation are folded into the weights at load, so a layer is six
// dispatches: QKV (with RMSNorm), attention (with QK-norm and RoPE), O (with
// the residual add), gate/up (with RMSNorm and SwiGLU), the online Hadamard
// of down's input, and down (with the residual add).
//
// GEMV kernels stream weights straight from memory and read activations
// through the cache; no threadgroup staging or barriers sit in the weight
// loop. Kernels that update the residual stream also write per-threadgroup
// partial sums of its squares, from which the next RMSNorm takes its factor.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SG = 8; // simdgroups per GEMV threadgroup

struct GemvArgs {
	uint K, N;
	float eps;
	uint parts; // partial sums of squares to read (norm kernels)
};

enum { PRO_PLAIN, PRO_NORM };
enum { EPI_STORE, EPI_ADD, EPI_SWIGLU };

// rowsPerSimdgroup is the number of weight rows each simdgroup streams.
constexpr uint rowsPerSimdgroup(int bits) { return bits == 8 ? 2 : 4; }

// gemv computes y = W·x for quantized rows W[N][K].
// BITS 8: int8 rows with a per-row FP32 scale; each lane takes 16 values
// per step. BITS 4: blocks of 32 values as 16 bytes (value j in the low
// nibble of byte j, value j+16 in the high nibble, both stored plus 8) with
// one FP16 scale per block; each lane takes one block per step.
template <int PRO, int EPI, int BITS>
inline void gemv(device const uchar *W, device const void *scale, device const float *x,
		device float *y, device const float *partsIn, device float *partsOut,
		constant GemvArgs &a, threadgroup float *tgPart, uint tg, uint sg, uint lane) {
	constexpr uint step = BITS == 8 ? 16 : 32;
	constexpr uint RPS = rowsPerSimdgroup(BITS);
	float inv = 1;
	if (PRO == PRO_NORM) {
		float s = 0;
		for (uint i = lane; i < a.parts; i += 32)
			s += partsIn[i];
		inv = rsqrt(simd_sum(s) / a.K + a.eps);
	}
	uint row0 = (tg * SG + sg) * RPS;
	float acc[RPS] = {0};
	if (BITS == 8) {
		device const uchar *wr = W + (ulong)row0 * a.K;
		for (uint i = lane * step; i < a.K; i += 32 * step) {
			device const float4 *xv = (device const float4 *)(x + i);
			float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
			for (uint r = 0; r < RPS; r++) {
				uint4 w = *(device const uint4 *)(wr + (ulong)r * a.K + i);
				acc[r] += dot(float4(as_type<char4>(w.x)), x0) + dot(float4(as_type<char4>(w.y)), x1) +
					dot(float4(as_type<char4>(w.z)), x2) + dot(float4(as_type<char4>(w.w)), x3);
			}
		}
	} else {
		device const uchar *wr = W + (ulong)row0 * (a.K / 2);
		device const half *sr = (device const half *)scale + (ulong)row0 * (a.K / 32);
		for (uint i = lane * step; i < a.K; i += 32 * step) {
			device const float4 *xv = (device const float4 *)(x + i);
			float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
			float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
			// The +8 offset is removed through the block sum of x.
			float sx = dot(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7, float4(1)) * 8;
			for (uint r = 0; r < RPS; r++) {
				uint4 w = *(device const uint4 *)(wr + (ulong)r * (a.K / 2) + i / 2);
				float d = float(sr[(ulong)r * (a.K / 32) + i / 32]);
				float4 t = float4(as_type<uchar4>(w.x & 0x0F0F0F0Fu)) * x0;
				t = fma(float4(as_type<uchar4>(w.y & 0x0F0F0F0Fu)), x1, t);
				t = fma(float4(as_type<uchar4>(w.z & 0x0F0F0F0Fu)), x2, t);
				t = fma(float4(as_type<uchar4>(w.w & 0x0F0F0F0Fu)), x3, t);
				t = fma(float4(as_type<uchar4>((w.x >> 4) & 0x0F0F0F0Fu)), x4, t);
				t = fma(float4(as_type<uchar4>((w.y >> 4) & 0x0F0F0F0Fu)), x5, t);
				t = fma(float4(as_type<uchar4>((w.z >> 4) & 0x0F0F0F0Fu)), x6, t);
				t = fma(float4(as_type<uchar4>((w.w >> 4) & 0x0F0F0F0Fu)), x7, t);
				acc[r] = fma(d, dot(t, float4(1)) - sx, acc[r]);
			}
		}
	}
	for (uint r = 0; r < RPS; r++) {
		acc[r] = simd_sum(acc[r]) * inv;
		if (BITS == 8)
			acc[r] *= ((device const float *)scale)[row0 + r];
	}
	if (EPI == EPI_SWIGLU) {
		if (lane == 0)
			for (uint r = 0; r < RPS; r += 2) {
				float g = acc[r];
				y[(row0 + r) / 2] = g / (1 + exp(-g)) * acc[r + 1];
			}
		return;
	}
	if (EPI == EPI_STORE) {
		if (lane == 0)
			for (uint r = 0; r < RPS; r++)
				y[row0 + r] = acc[r];
		return;
	}
	// EPI_ADD: update the residual and publish this threadgroup's sum of
	// squares of the updated rows.
	float ss = 0;
	if (lane == 0)
		for (uint r = 0; r < RPS; r++) {
			float v = y[row0 + r] + acc[r];
			y[row0 + r] = v;
			ss += v * v;
		}
	if (lane == 0)
		tgPart[sg] = ss;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (sg == 0 && lane == 0) {
		float t = 0;
		for (uint i = 0; i < SG; i++)
			t += tgPart[i];
		partsOut[tg] = t;
	}
}

#define GEMV_KERNEL(name, PRO, EPI, BITS)                                                        \
	kernel void name(device const uchar *W [[buffer(0)]], device const void *scale [[buffer(1)]],   \
			device const float *x [[buffer(2)]], device float *y [[buffer(3)]],                    \
			device const float *partsIn [[buffer(4)]], constant GemvArgs &a [[buffer(5)]],         \
			device float *partsOut [[buffer(6)]],                                                  \
			uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],  \
			uint lane [[thread_index_in_simdgroup]]) {                                             \
		threadgroup float tgPart[SG];                                                            \
		gemv<PRO, EPI, BITS>(W, scale, x, y, partsIn, partsOut, a, tgPart, tg, sg, lane);        \
	}

GEMV_KERNEL(gemv_qkv, PRO_NORM, EPI_STORE, 8)
GEMV_KERNEL(gemv_o, PRO_PLAIN, EPI_ADD, 8)
GEMV_KERNEL(gemv_gateup, PRO_NORM, EPI_SWIGLU, 8)
GEMV_KERNEL(gemv_down, PRO_PLAIN, EPI_ADD, 8)
GEMV_KERNEL(gemv_qkv_q4, PRO_NORM, EPI_STORE, 4)
GEMV_KERNEL(gemv_o_q4, PRO_PLAIN, EPI_ADD, 4)
GEMV_KERNEL(gemv_gateup_q4, PRO_NORM, EPI_SWIGLU, 4)
GEMV_KERNEL(gemv_down_q4, PRO_PLAIN, EPI_ADD, 4)

// rotate4096 replaces each 4096-value block of x with H·diag(signs)·x
// (signs carry the normalization); threadgroup b handles block b.
kernel void rotate4096(device float *x [[buffer(0)]], device const float *signs [[buffer(1)]],
		uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	constexpr uint N = 4096, T = 1024;
	threadgroup float v[N];
	device float *xb = x + b * N;
	device const float *sb = signs + b * N;
	// Each thread owns 4 consecutive values: the first two butterfly levels
	// run in registers.
	float4 r = *(device const float4 *)(xb + tid * 4) * *(device const float4 *)(sb + tid * 4);
	r = float4(r.x + r.y, r.x - r.y, r.z + r.w, r.z - r.w);
	r = float4(r.x + r.z, r.y + r.w, r.x - r.z, r.y - r.w);
	*(threadgroup float4 *)(v + tid * 4) = r;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint h = 4; h < N; h <<= 1) {
		for (uint t = 0; t < 2; t++) {
			uint idx = t * T + tid;
			uint i = (idx / h) * 2 * h + (idx % h);
			float p = v[i], q = v[i + h];
			v[i] = p + q;
			v[i + h] = p - q;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	*(device float4 *)(xb + tid * 4) = *(threadgroup float4 *)(v + tid * 4);
}

struct AttnArgs {
	uint pos;      // this token's position; keys 0..pos are attended
	uint ropeSin;  // offset of the sine table in rope
	float eps, scale;
};

// normRope applies the head RMSNorm and RoPE to one 128-value head held as
// lanes' dims (l, l+32, l+64, l+96).
inline float4 normRope(float4 v, device const float *w, device const float *rope, constant AttnArgs &a, uint lane) {
	float ss = simd_sum(dot(v, v));
	float inv = rsqrt(ss / 128 + a.eps);
	v *= inv * float4(w[lane], w[lane + 32], w[lane + 64], w[lane + 96]);
	uint base = a.pos * 64;
	float c0 = rope[base + lane], c1 = rope[base + lane + 32];
	float s0 = rope[a.ropeSin + base + lane], s1 = rope[a.ropeSin + base + lane + 32];
	return float4(v.x * c0 - v.z * s0, v.y * c1 - v.w * s1, v.z * c0 + v.x * s0, v.w * c1 + v.y * s1);
}

// attend1 handles one new token: threadgroup g is KV head g, and its four
// simdgroups are the query heads sharing it. The token's key and value are
// appended to the cache, then each query head attends to keys 0..pos with a
// streaming softmax.
kernel void attend1(device const float *qkv [[buffer(0)]], device float *kc [[buffer(1)]],
		device float *vc [[buffer(2)]], device const float *qn [[buffer(3)]],
		device const float *kn [[buffer(4)]], device const float *rope [[buffer(5)]],
		device float *ctx [[buffer(6)]], constant AttnArgs &a [[buffer(7)]],
		uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
		uint lane [[thread_index_in_simdgroup]]) {
	if (sg == 0) {
		device const float *k = qkv + 4096 + g * 128;
		device const float *v = qkv + 5120 + g * 128;
		float4 kv = normRope(float4(k[lane], k[lane + 32], k[lane + 64], k[lane + 96]), kn, rope, a, lane);
		device float *kd = kc + a.pos * 1024 + g * 128;
		device float *vd = vc + a.pos * 1024 + g * 128;
		kd[lane] = kv.x, kd[lane + 32] = kv.y, kd[lane + 64] = kv.z, kd[lane + 96] = kv.w;
		vd[lane] = v[lane], vd[lane + 32] = v[lane + 32], vd[lane + 64] = v[lane + 64], vd[lane + 96] = v[lane + 96];
	}
	threadgroup_barrier(mem_flags::mem_device);
	uint head = g * 4 + sg;
	device const float *q = qkv + head * 128;
	float4 qv = normRope(float4(q[lane], q[lane + 32], q[lane + 64], q[lane + 96]), qn, rope, a, lane) * a.scale;
	float m = -INFINITY, l = 0;
	float4 acc = 0;
	for (uint j = 0; j <= a.pos; j++) {
		device const float *k = kc + j * 1024 + g * 128;
		float s = simd_sum(dot(qv, float4(k[lane], k[lane + 32], k[lane + 64], k[lane + 96])));
		float mn = max(m, s);
		float c = exp(m - mn), p = exp(s - mn);
		device const float *v = vc + j * 1024 + g * 128;
		acc = acc * c + p * float4(v[lane], v[lane + 32], v[lane + 64], v[lane + 96]);
		l = l * c + p;
		m = mn;
	}
	acc /= l;
	device float *o = ctx + head * 128;
	o[lane] = acc.x, o[lane + 32] = acc.y, o[lane + 64] = acc.z, o[lane + 96] = acc.w;
}
