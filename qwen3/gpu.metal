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
// (signs carry the normalization); threadgroup b handles block b, and rows
// hold perRow blocks.
kernel void rotate4096(device float *x [[buffer(0)]], device const float *signs [[buffer(1)]],
		constant uint &perRow [[buffer(2)]],
		uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	constexpr uint N = 4096, T = 1024;
	threadgroup float v[N];
	device float *xb = x + b * N;
	device const float *sb = signs + (b % perRow) * N;
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

// normRopeAt applies the head RMSNorm and RoPE to one 128-value head held as
// lanes' dims (l, l+32, l+64, l+96).
inline float4 normRopeAt(float4 v, device const float *w, device const float *rope, AttnArgs a, uint lane) {
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
		float4 kv = normRopeAt(float4(k[lane], k[lane + 32], k[lane + 64], k[lane + 96]), kn, rope, a, lane);
		device float *kd = kc + a.pos * 1024 + g * 128;
		device float *vd = vc + a.pos * 1024 + g * 128;
		kd[lane] = kv.x, kd[lane + 32] = kv.y, kd[lane + 64] = kv.z, kd[lane + 96] = kv.w;
		vd[lane] = v[lane], vd[lane + 32] = v[lane + 32], vd[lane + 64] = v[lane + 64], vd[lane + 96] = v[lane + 96];
	}
	threadgroup_barrier(mem_flags::mem_device);
	uint head = g * 4 + sg;
	device const float *q = qkv + head * 128;
	float4 qv = normRopeAt(float4(q[lane], q[lane + 32], q[lane + 64], q[lane + 96]), qn, rope, a, lane) * a.scale;
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

// ---- Batched (multi-token) kernels ----

struct MMArgs {
	uint K, N, M;
	float eps;
	uint parts;    // partial sums of squares per row to read (norm kernels)
	uint partsOut; // partial sums per row written (N / MM_BN, residual kernels)
	uint splitK;   // K values per split; splits > 1 write raw sums to scratch
	uint splits;
	uint padM;     // M rounded up to the tile height (scratch row count)
};

constant constexpr uint MM_BN = 64, MM_BK = 32;

// mm computes y[M][N] = x[M][K]·Wᵀ with simdgroup matrices. A threadgroup of
// T threads owns BM tokens by 64 weight rows, and each simdgroup a 16×16
// output block; each K step dequantizes a 64×32 weight tile to FP16 (int8
// codes exactly, 4-bit codes times their FP16 scale) and rounds the BM×32
// activation tile to FP16, so every weight is read once per BM tokens.
// Products accumulate in FP32.
template <int PRO, int EPI, int BITS, uint BM, uint T>
inline void mm(device const uchar *W, device const void *scale, device const float *x, device float *y,
		device const float *partsIn, device float *partsOut, constant MMArgs &a, device float *scratch,
		threadgroup float *smem, threadgroup float *inv, uint3 tg, uint tid, uint sg) {
	constexpr uint WPER = MM_BN * MM_BK / T; // weight values per thread (16 or 8)
	constexpr uint XPER = BM * MM_BK / T;    // activation values per thread (4)
	threadgroup half *Ws = (threadgroup half *)smem;                 // [64][32]
	threadgroup half *Xs = (threadgroup half *)smem + MM_BN * MM_BK; // [BM][32]
	uint n0 = tg.x * MM_BN, m0 = tg.y * BM;
	if (tid < BM) {
		float f = 1;
		uint m = m0 + tid;
		if (PRO == PRO_NORM && m < a.M) {
			float s = 0;
			for (uint i = 0; i < a.parts; i++)
				s += partsIn[m * a.parts + i];
			f = rsqrt(s / a.K + a.eps);
		}
		inv[tid] = f;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	uint sn = (sg % 4) * 16, sm = (sg / 4) * 16;
	simdgroup_float8x8 acc[2][2];
	for (uint i = 0; i < 2; i++)
		acc[i][0] = acc[i][1] = make_filled_simdgroup_matrix<float, 8, 8>(0);
	uint wr = tid / (MM_BK / WPER), wc = (tid % (MM_BK / WPER)) * WPER;
	uint xr = tid / (MM_BK / XPER), xc = (tid % (MM_BK / XPER)) * XPER;
	bool live = m0 + xr < a.M;
	float f = inv[xr];
	device const float *xrow = x + (ulong)(m0 + xr) * a.K + xc;
	device const uchar *wrow = BITS == 8 ? W + (ulong)(n0 + wr) * a.K + wc : W + (ulong)(n0 + wr) * (a.K / 2) + (wc % 16);
	device const half *srow = (device const half *)scale + (ulong)(n0 + wr) * (a.K / 32);
	// Global loads for the next step are issued before this step computes.
	uint4 wreg = 0;
	half d = 0;
	float4 xreg = 0;
#define MM_FETCH(k0)                                                        \
	if (BITS == 8) {                                                        \
		if (WPER == 16)                                                     \
			wreg = *(device const uint4 *)(wrow + (k0));                    \
		else                                                                \
			wreg.xy = *(device const uint2 *)(wrow + (k0));                 \
	} else {                                                                \
		if (WPER == 16)                                                     \
			wreg = *(device const uint4 *)(wrow + (k0) / 2);                \
		else                                                                \
			wreg.xy = *(device const uint2 *)(wrow + (k0) / 2);             \
		d = srow[(k0) / 32];                                                \
	}                                                                       \
	xreg = live ? *(device const float4 *)(xrow + (k0)) * f : 0;
	uint kBeg = tg.z * a.splitK, kEnd = kBeg + a.splitK;
	MM_FETCH(kBeg)
	for (uint k0 = kBeg; k0 < kEnd; k0 += MM_BK) {
		threadgroup half *wd = Ws + wr * MM_BK + wc;
		if (BITS == 8) {
			*(threadgroup half4 *)wd = half4(as_type<char4>(wreg.x));
			*(threadgroup half4 *)(wd + 4) = half4(as_type<char4>(wreg.y));
			if (WPER == 16) {
				*(threadgroup half4 *)(wd + 8) = half4(as_type<char4>(wreg.z));
				*(threadgroup half4 *)(wd + 12) = half4(as_type<char4>(wreg.w));
			}
		} else {
			uint4 q = (wc >= 16 ? wreg >> 4 : wreg) & 0x0F0F0F0Fu;
			*(threadgroup half4 *)wd = (half4(as_type<uchar4>(q.x)) - 8) * d;
			*(threadgroup half4 *)(wd + 4) = (half4(as_type<uchar4>(q.y)) - 8) * d;
			if (WPER == 16) {
				*(threadgroup half4 *)(wd + 8) = (half4(as_type<uchar4>(q.z)) - 8) * d;
				*(threadgroup half4 *)(wd + 12) = (half4(as_type<uchar4>(q.w)) - 8) * d;
			}
		}
		*(threadgroup half4 *)(Xs + xr * MM_BK + xc) = half4(xreg);
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (k0 + MM_BK < kEnd) {
			MM_FETCH(k0 + MM_BK)
		}
		for (uint k8 = 0; k8 < MM_BK; k8 += 8) {
			simdgroup_half8x8 A0, A1, B0, B1;
			simdgroup_load(A0, Xs + sm * MM_BK + k8, MM_BK);
			simdgroup_load(A1, Xs + (sm + 8) * MM_BK + k8, MM_BK);
			simdgroup_load(B0, Ws + sn * MM_BK + k8, MM_BK, ulong2(0, 0), true);
			simdgroup_load(B1, Ws + (sn + 8) * MM_BK + k8, MM_BK, ulong2(0, 0), true);
			simdgroup_multiply_accumulate(acc[0][0], A0, B0, acc[0][0]);
			simdgroup_multiply_accumulate(acc[0][1], A1, B0, acc[0][1]);
			simdgroup_multiply_accumulate(acc[1][0], A0, B1, acc[1][0]);
			simdgroup_multiply_accumulate(acc[1][1], A1, B1, acc[1][1]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
#undef MM_FETCH
	if (a.splits > 1) {
		// Raw sums; mm_finish adds the splits and applies the epilogue.
		device float *out = scratch + ((ulong)tg.z * a.padM + m0) * a.N + n0;
		for (uint nb = 0; nb < 2; nb++) {
			simdgroup_store(acc[nb][0], out + sm * a.N + sn + nb * 8, a.N);
			simdgroup_store(acc[nb][1], out + (sm + 8) * a.N + sn + nb * 8, a.N);
		}
		return;
	}
	threadgroup float *Cs = smem; // [BM][64], reusing the tiles
	for (uint nb = 0; nb < 2; nb++) {
		simdgroup_store(acc[nb][0], Cs + sm * MM_BN + sn + nb * 8, MM_BN);
		simdgroup_store(acc[nb][1], Cs + (sm + 8) * MM_BN + sn + nb * 8, MM_BN);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint e = tid; e < BM * MM_BN; e += T) {
		uint m = e / MM_BN, n = e % MM_BN;
		float v = Cs[e];
		if (BITS == 8)
			v *= ((device const float *)scale)[n0 + n];
		if (m0 + m >= a.M)
			continue;
		if (EPI == EPI_STORE) {
			y[(ulong)(m0 + m) * a.N + n0 + n] = v;
		} else if (EPI == EPI_ADD) {
			device float *p = y + (ulong)(m0 + m) * a.N + n0 + n;
			v += *p;
			*p = v;
			Cs[e] = v;
		} else {
			Cs[e] = v; // SwiGLU pairs are combined below
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (EPI == EPI_SWIGLU) {
		for (uint e = tid; e < BM * MM_BN / 2; e += T) {
			uint m = e / (MM_BN / 2), j = e % (MM_BN / 2);
			if (m0 + m >= a.M)
				continue;
			float g = Cs[m * MM_BN + 2 * j], u = Cs[m * MM_BN + 2 * j + 1];
			y[(ulong)(m0 + m) * (a.N / 2) + n0 / 2 + j] = g / (1 + exp(-g)) * u;
		}
	}
	if (EPI == EPI_ADD && tid < BM && m0 + tid < a.M) {
		float ss = 0;
		for (uint n = 0; n < MM_BN; n++)
			ss += Cs[tid * MM_BN + n] * Cs[tid * MM_BN + n];
		partsOut[(m0 + tid) * a.partsOut + tg.x] = ss;
	}
}

#define MM_KERNEL(name, PRO, EPI, BITS, BM)                                                      \
	kernel void name(device const uchar *W [[buffer(0)]], device const void *scale [[buffer(1)]],   \
			device const float *x [[buffer(2)]], device float *y [[buffer(3)]],                    \
			device const float *partsIn [[buffer(4)]], constant MMArgs &a [[buffer(5)]],           \
			device float *partsOut [[buffer(6)]], device float *scratch [[buffer(7)]],             \
			uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],   \
			uint sg [[simdgroup_index_in_threadgroup]]) {                                          \
		threadgroup float smem[BM * MM_BN];                                                      \
		threadgroup float inv[BM];                                                               \
		mm<PRO, EPI, BITS, BM, BM * 8>(W, scale, x, y, partsIn, partsOut, a, scratch, smem, inv, tg, tid, sg); \
	}

// mm_finish adds split-K sums for row m and 64 columns and applies the
// epilogue: row scale (int8), store, residual add with the sum of squares,
// or SwiGLU. Threadgroup (column tile, m) has 64 threads.
template <int EPI, int BITS>
inline void mmFinish(device const void *scale, device float *y, device float *partsOut, constant MMArgs &a,
		device const float *scratch, threadgroup float *tgPart, uint2 tg, uint tid, uint sg, uint lane) {
	uint n = tg.x * MM_BN + tid, m = tg.y;
	float v = 0;
	for (uint s = 0; s < a.splits; s++)
		v += scratch[((ulong)s * a.padM + m) * a.N + n];
	if (BITS == 8)
		v *= ((device const float *)scale)[n];
	if (EPI == EPI_STORE) {
		y[(ulong)m * a.N + n] = v;
	} else if (EPI == EPI_SWIGLU) {
		float u = simd_shuffle_down(v, 1);
		if (tid % 2 == 0)
			y[(ulong)m * (a.N / 2) + n / 2] = v / (1 + exp(-v)) * u;
	} else {
		device float *p = y + (ulong)m * a.N + n;
		v += *p;
		*p = v;
		float ss = simd_sum(v * v);
		if (lane == 0)
			tgPart[sg] = ss;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid == 0)
			partsOut[m * a.partsOut + tg.x] = tgPart[0] + tgPart[1];
	}
}

#define MM_FINISH(name, EPI, BITS)                                                                   \
	kernel void name(device const void *scale [[buffer(1)]], device float *y [[buffer(3)]],             \
			constant MMArgs &a [[buffer(5)]], device float *partsOut [[buffer(6)]],                   \
			device const float *scratch [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],      \
			uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],     \
			uint lane [[thread_index_in_simdgroup]]) {                                                 \
		threadgroup float tgPart[2];                                                                 \
		mmFinish<EPI, BITS>(scale, y, partsOut, a, scratch, tgPart, tg, tid, sg, lane);              \
	}

MM_FINISH(mm_finish_store, EPI_STORE, 8)
MM_FINISH(mm_finish_add, EPI_ADD, 8)
MM_FINISH(mm_finish_swiglu, EPI_SWIGLU, 8)
MM_FINISH(mm_finish_store_q4, EPI_STORE, 4)
MM_FINISH(mm_finish_add_q4, EPI_ADD, 4)
MM_FINISH(mm_finish_swiglu_q4, EPI_SWIGLU, 4)

#define MM_KERNELS(suffix, BITS, BM)                                  \
	MM_KERNEL(mm_qkv##suffix, PRO_NORM, EPI_STORE, BITS, BM)          \
	MM_KERNEL(mm_o##suffix, PRO_PLAIN, EPI_ADD, BITS, BM)             \
	MM_KERNEL(mm_gateup##suffix, PRO_NORM, EPI_SWIGLU, BITS, BM)      \
	MM_KERNEL(mm_down##suffix, PRO_PLAIN, EPI_ADD, BITS, BM)

MM_KERNELS(, 8, 32)
MM_KERNELS(_q4, 4, 32)
MM_KERNELS(_16, 8, 16)
MM_KERNELS(_q4_16, 4, 16)

// qkRope prepares M new token rows of possibly several sequences; row m
// has position info[m].x. Query heads are normalized, rotated, and scaled in
// place; key heads are normalized, rotated, and stored in cache row m; value
// heads are copied to cache row m. Threadgroup (h, m) is one simdgroup for
// head h of row m.
kernel void qkRope(device float *qkv [[buffer(0)]], device float *kc [[buffer(1)]],
		device float *vc [[buffer(2)]], device const float *qn [[buffer(3)]],
		device const float *kn [[buffer(4)]], device const float *rope [[buffer(5)]],
		device const uint2 *info [[buffer(6)]], constant AttnArgs &a0 [[buffer(7)]],
		uint2 hm [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
	uint h = hm.x, m = hm.y;
	AttnArgs a = a0;
	a.pos = info[m].x;
	device float *row = qkv + m * 6144;
	if (h < 40) {
		bool isQ = h < 32;
		device float *src = row + h * 128;
		float4 v = float4(src[lane], src[lane + 32], src[lane + 64], src[lane + 96]);
		v = normRopeAt(v, isQ ? qn : kn, rope, a, lane);
		device float *d = isQ ? src : kc + m * 1024 + (h - 32) * 128;
		if (isQ)
			v *= a.scale;
		d[lane] = v.x, d[lane + 32] = v.y, d[lane + 64] = v.z, d[lane + 96] = v.w;
		return;
	}
	device const float *src = row + 5120 + (h - 40) * 128;
	device float *d = vc + m * 1024 + (h - 40) * 128;
	d[lane] = src[lane], d[lane + 32] = src[lane + 32], d[lane + 64] = src[lane + 64], d[lane + 96] = src[lane + 96];
}

// attendM attends row m to cache rows info[m].y..m, its own sequence's
// earlier tokens. Threadgroup (g, m) holds the four query heads of KV head g
// as simdgroups; queries were prepared by qkRope.
kernel void attendM(device const float *qkv [[buffer(0)]], device const float *kc [[buffer(1)]],
		device const float *vc [[buffer(2)]], device const uint2 *info [[buffer(3)]],
		device float *ctx [[buffer(6)]],
		uint2 gm [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
		uint lane [[thread_index_in_simdgroup]]) {
	uint g = gm.x, m = gm.y;
	uint head = g * 4 + sg;
	device const float *q = qkv + m * 6144 + head * 128;
	float4 qv = float4(q[lane], q[lane + 32], q[lane + 64], q[lane + 96]);
	float mx = -INFINITY, l = 0;
	float4 acc = 0;
	for (uint j = info[m].y; j <= m; j++) {
		device const float *k = kc + j * 1024 + g * 128;
		float s = simd_sum(dot(qv, float4(k[lane], k[lane + 32], k[lane + 64], k[lane + 96])));
		float mn = max(mx, s);
		float c = exp(mx - mn), p = exp(s - mn);
		device const float *v = vc + j * 1024 + g * 128;
		acc = acc * c + p * float4(v[lane], v[lane + 32], v[lane + 64], v[lane + 96]);
		l = l * c + p;
		mx = mn;
	}
	acc /= l;
	device float *o = ctx + m * 4096 + head * 128;
	o[lane] = acc.x, o[lane + 32] = acc.y, o[lane + 64] = acc.z, o[lane + 96] = acc.w;
}
