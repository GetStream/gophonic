// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// Qwen3-ASR audio encoder kernels for Apple GPUs. Activations are FP32 and
// time-major with channels innermost, as in the CPU encoder. Weights are FP16
// rows scaled by a power of two per row, which holds every BF16 checkpoint
// weight exactly; products accumulate in FP32 simdgroup matrices.

#include <metal_stdlib>
using namespace metal;

// geluExact is x·Φ(x), PyTorch's default erf-based GELU, evaluated as the
// CPU kernels do: the normal survival function Q and density φ at the
// nearest 1/128 grid point r (table[i] = {Q(i/128), φ(i/128)}), then
// Q(r+d) = Q(r) - φ(r)·d·(1 - r·d/2 + (r²-1)·d²/6), within 2e-7 absolute.
inline float geluExact(float x, constant float2 *table) {
	float a = fabs(x);
	if (!(a < 8))
		return x > 0 ? x : (x == x ? -0.0f : x);
	uint i = uint(a * 128 + 0.5f);
	float r = float(i) * (1.0f / 128), d = a - r;
	float2 t = table[i];
	float q = t.x - t.y * d * (1 - d * (r * 0.5f - d * ((r * r - 1) * (1.0f / 6))));
	return x * (x > 0 ? 1 - q : q);
}

// ---- Matrix products ----

enum { EPI_STORE, EPI_BIAS, EPI_BIAS_GELU, EPI_RESIDUAL };

struct GemmArgs {
	uint M, N, K;
	uint splitK, splits; // K per split; splits > 1 write raw sums to scratch
	uint padN;           // scratch row width: N rounded up to the tile
};

struct ConvArgs {
	uint frames;         // feature frames of the whole input
	uint chunkFrames;    // frames per chunk before padding
	uint inTime, inFreq; // input extents per chunk
	uint outTime, outFreq;
	uint ch;             // channels
	uint chunk0;         // conv1: the first chunk of this dispatch
};

constant constexpr uint BM = 64, BN = 64, BK = 32, GT = 128; // tile, threads

// epilogue scales one raw sum and writes it with the kernel's epilogue.
template <int EPI>
inline void epilogue(float v, uint m, uint n, device const float *scale, device const float *bias,
		device float *y, constant GemmArgs &a, constant float2 *table) {
	v *= scale[n];
	device float *out = y + (ulong)m * a.N + n;
	if (EPI == EPI_BIAS)
		v += bias[n];
	else if (EPI == EPI_BIAS_GELU)
		v = geluExact(v + bias[n], table);
	else if (EPI == EPI_RESIDUAL)
		v = *out + (v + bias[n]);
	*out = v;
}

// gemm computes y[M][N] = x[M][K]·(W[N][K]·diag(scale))ᵀ with an epilogue:
// store, add bias, add bias then GELU, or add bias to the residual y. K is
// a multiple of BK; M and N are arbitrary. A threadgroup of four simdgroups
// owns a 64×64 output tile, each simdgroup 32×32 as 4×4 8×8 matrices. Each
// K step stages a 64×32 tile of activations (rounded to FP16) and of
// weights in threadgroup memory; the next step's global loads are issued
// before this step's products, and products accumulate in FP32. Small
// products split K across threadgroups (tg.z): each writes its raw sums to
// scratch, and gemm_finish adds them and applies the epilogue.
//
// With CONV, x is the [chunk][time][freq][ch] input of a 3×3 stride-2
// convolution and row m of the product is output (chunk, t, f): its K values
// are, for each time tap, the three frequency taps of every channel, read
// in place (zero outside the input), so no im2col matrix is stored.
template <int EPI, bool CONV>
inline void gemm(device const float *x, device const half *W, device const float *scale,
		device const float *bias, device float *y, constant GemmArgs &a, constant ConvArgs &cv,
		constant float2 *table, device float *scratch, threadgroup float *smem, uint3 tg, uint tid, uint sg) {
	threadgroup half *As = (threadgroup half *)smem, *Bs = As + BM * BK;
	uint m0 = tg.y * BM, n0 = tg.x * BN;
	uint kBeg = tg.z * a.splitK, kEnd = kBeg + a.splitK;
	uint sm = (sg / 2) * 32, sn = (sg % 2) * 32;
	simdgroup_float8x8 acc[4][4];
	for (uint i = 0; i < 4; i++)
		for (uint j = 0; j < 4; j++)
			acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0);
	// Each thread stages 16 consecutive values of one row of each tile.
	uint lr = tid / 2, lc = (tid % 2) * 16;
	bool rowA = m0 + lr < a.M, rowB = n0 + lr < a.N;
	device const float *xa = x + (ulong)(m0 + lr) * a.K + lc;
	device const half *wb = W + (ulong)(n0 + lr) * a.K + lc;
	uint cb = 0, ct = 0, cf = 0; // this thread's output (chunk, t, f)
	if (CONV) {
		uint m = m0 + lr, per = cv.outTime * cv.outFreq;
		cb = m / per, ct = m % per / cv.outFreq, cf = m % cv.outFreq;
	}
	float4 a0, a1, a2, a3;
	half4 b0, b1, b2, b3;
	auto fetch = [&](uint k0) {
		a0 = a1 = a2 = a3 = 0;
		b0 = b1 = b2 = b3 = 0;
		if (rowA) {
			device const float4 *p = (device const float4 *)(xa + k0);
			if (CONV) {
				uint k = k0 + lc, tj = k / (3 * cv.ch), rest = k % (3 * cv.ch), fi = rest / cv.ch, c = rest % cv.ch;
				int ti = int(2 * ct + tj) - 1, fb = int(2 * cf + fi) - 1;
				p = ti >= 0 && uint(ti) < cv.inTime && fb >= 0 && uint(fb) < cv.inFreq
					? (device const float4 *)(x + (((ulong)cb * cv.inTime + ti) * cv.inFreq + fb) * cv.ch + c)
					: nullptr;
			}
			if (p != nullptr)
				a0 = p[0], a1 = p[1], a2 = p[2], a3 = p[3];
		}
		if (rowB) {
			device const half4 *p = (device const half4 *)(wb + k0);
			b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3];
		}
	};
	fetch(kBeg);
	for (uint k0 = kBeg; k0 < kEnd; k0 += BK) {
		threadgroup half4 *ad = (threadgroup half4 *)(As + lr * BK + lc);
		ad[0] = half4(a0), ad[1] = half4(a1), ad[2] = half4(a2), ad[3] = half4(a3);
		threadgroup half4 *bd = (threadgroup half4 *)(Bs + lr * BK + lc);
		bd[0] = b0, bd[1] = b1, bd[2] = b2, bd[3] = b3;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (k0 + BK < kEnd)
			fetch(k0 + BK);
		for (uint k8 = 0; k8 < BK; k8 += 8) {
			simdgroup_half8x8 A[4], B[4];
			for (uint i = 0; i < 4; i++) {
				simdgroup_load(A[i], As + (sm + 8 * i) * BK + k8, BK);
				simdgroup_load(B[i], Bs + (sn + 8 * i) * BK + k8, BK, ulong2(0, 0), true);
			}
			for (uint i = 0; i < 4; i++)
				for (uint j = 0; j < 4; j++)
					simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	if (a.splits > 1) {
		device float *out = scratch + ((ulong)tg.z * (a.M + BM) + m0 + sm) * a.padN + n0 + sn;
		for (uint i = 0; i < 4; i++)
			for (uint j = 0; j < 4; j++)
				simdgroup_store(acc[i][j], out + 8 * i * a.padN + 8 * j, a.padN);
		return;
	}
	// Stage the 64×64 output tile, then apply the epilogue with bounds
	// checks.
	threadgroup float *Cs = smem;
	for (uint i = 0; i < 4; i++)
		for (uint j = 0; j < 4; j++)
			simdgroup_store(acc[i][j], Cs + (sm + 8 * i) * BN + sn + 8 * j, BN);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint e = tid; e < BM * BN; e += GT) {
		uint m = m0 + e / BN, n = n0 + e % BN;
		if (m >= a.M || n >= a.N)
			continue;
		epilogue<EPI>(Cs[e], m, n, scale, bias, y, a, table);
	}
}


#define GEMM_KERNEL(name, EPI, CONV)                                                               \
	kernel void name(device const float *x [[buffer(0)]], device const half *W [[buffer(1)]],      \
			device const float *scale [[buffer(2)]], device const float *bias [[buffer(3)]],      \
			device float *y [[buffer(4)]], constant GemmArgs &a [[buffer(5)]],                    \
			constant ConvArgs &cv [[buffer(6)]], constant float2 *table [[buffer(7)]],            \
			device float *scratch [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],       \
			uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
		threadgroup float smem[BM * BN];                                                         \
		gemm<EPI, CONV>(x, W, scale, bias, y, a, cv, table, scratch, smem, tg, tid, sg);        \
	}

// gemm_finish adds the split sums of element (m, n) and applies the
// epilogue; the grid is one thread per element of the padded product.
#define GEMM_FINISH(name, EPI)                                                                     \
	kernel void name(device const float *scale [[buffer(2)]], device const float *bias [[buffer(3)]], \
			device float *y [[buffer(4)]], constant GemmArgs &a [[buffer(5)]],                    \
			constant float2 *table [[buffer(7)]], device const float *scratch [[buffer(8)]],      \
			uint2 id [[thread_position_in_grid]]) {                                              \
		uint n = id.x, m = id.y;                                                                 \
		if (m >= a.M || n >= a.N)                                                                \
			return;                                                                              \
		float v = 0;                                                                             \
		for (uint s = 0; s < a.splits; s++)                                                      \
			v += scratch[((ulong)s * (a.M + BM) + m) * a.padN + n];                               \
		epilogue<EPI>(v, m, n, scale, bias, y, a, table);                                        \
	}

GEMM_FINISH(gemm_finish_store, EPI_STORE)
GEMM_FINISH(gemm_finish_bias, EPI_BIAS)
GEMM_FINISH(gemm_finish_bias_gelu, EPI_BIAS_GELU)
GEMM_FINISH(gemm_finish_residual, EPI_RESIDUAL)

GEMM_KERNEL(gemm_store, EPI_STORE, false)
GEMM_KERNEL(gemm_bias, EPI_BIAS, false)
GEMM_KERNEL(gemm_bias_gelu, EPI_BIAS_GELU, false)
GEMM_KERNEL(gemm_residual, EPI_RESIDUAL, false)
GEMM_KERNEL(conv_gelu, EPI_BIAS_GELU, true)

// ---- Convolution stem ----


// conv1 is the first 3×3 stride-2 convolution over one input channel with
// its bias and GELU, computed directly. Grid (ch, outTime·outFreq, chunk);
// frames past a chunk's real ones read as zero, as the reference pads.
kernel void conv1(device const float *mel [[buffer(0)]], device const float *w [[buffer(1)]],
		device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
		constant ConvArgs &a [[buffer(4)]], constant float2 *table [[buffer(5)]],
		uint3 id [[thread_position_in_grid]]) {
	uint c = id.x, p = id.y, chunk = id.z;
	if (c >= a.ch)
		return;
	uint t = p / a.outFreq, f = p % a.outFreq;
	uint start = (a.chunk0 + chunk) * a.chunkFrames, real = min(a.chunkFrames, a.frames - start);
	float s = bias[c];
	for (uint tj = 0; tj < 3; tj++) {
		int ti = int(2 * t + tj) - 1;
		if (ti < 0 || uint(ti) >= real)
			continue;
		for (uint fi = 0; fi < 3; fi++) {
			int fb = int(2 * f + fi) - 1;
			if (fb < 0 || uint(fb) >= a.inFreq)
				continue;
			s += w[c * 9 + tj * 3 + fi] * mel[(ulong)fb * a.frames + start + ti];
		}
	}
	y[((ulong)chunk * a.outTime * a.outFreq + p) * a.ch + c] = geluExact(s, table);
}

// gather copies the rows of real frames from the conv_out product and adds
// their sinusoidal positions: row i takes source row src[i].x at chunk
// position src[i].y. Grid (d/4, rows).
kernel void gather(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
		device const uint2 *src [[buffer(2)]], device const float *pos [[buffer(3)]],
		constant uint &d [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
	uint c = id.x * 4, i = id.y;
	if (c >= d)
		return;
	uint2 s = src[i];
	*(device float4 *)(y + (ulong)i * d + c) =
		*(device const float4 *)(x + (ulong)s.x * d + c) + *(device const float4 *)(pos + (ulong)s.y * d + c);
}

// ---- LayerNorm and attention ----

// layerNorm writes LayerNorm(x[r])·w+b (epsilon 1e-5) to y[r]; threadgroup
// r of 256 threads normalizes row r of width d, a multiple of 4.
kernel void layerNorm(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
		device const float *w [[buffer(2)]], device const float *b [[buffer(3)]],
		constant uint &d [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
		uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
		uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float part[8];
	device const float4 *row = (device const float4 *)(x + (ulong)r * d);
	float s = 0;
	for (uint i = tid; i < d / 4; i += 256)
		s += dot(row[i], float4(1));
	s = simd_sum(s);
	if (lane == 0)
		part[sg] = s;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float mean = 0;
	for (uint i = 0; i < 8; i++)
		mean += part[i];
	mean /= d;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float v = 0;
	for (uint i = tid; i < d / 4; i += 256) {
		float4 t = row[i] - mean;
		v += dot(t, t);
	}
	v = simd_sum(v);
	if (lane == 0)
		part[sg] = v;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float var = 0;
	for (uint i = 0; i < 8; i++)
		var += part[i];
	float inv = rsqrt(var / d + 1e-5f);
	device float4 *out = (device float4 *)(y + (ulong)r * d);
	for (uint i = tid; i < d / 4; i += 256)
		out[i] = (row[i] - mean) * inv * ((device const float4 *)w)[i] + ((device const float4 *)b)[i];
}

struct AttnArgs {
	uint rows, window, heads, d;
	float scale;
};

// attend is bidirectional self-attention within windows of a.window rows
// for 64-wide heads: simdgroup (head, row) attends from row to every row of
// its window with a streaming softmax over blocks of eight keys. qkv rows
// hold queries, keys, and values, each d wide. Grid (heads, rows/4) of four
// simdgroups.
kernel void attend(device const float *qkv [[buffer(0)]], device float *ctx [[buffer(1)]],
		constant AttnArgs &a [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
		uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
	constexpr uint AK = 8;
	uint h = tg.x, r = tg.y * 4 + sg;
	if (r >= a.rows)
		return;
	uint w0 = r / a.window * a.window, w1 = min(w0 + a.window, a.rows);
	uint stride = 3 * a.d;
	float2 q = *(device const float2 *)(qkv + (ulong)r * stride + h * 64 + 2 * lane) * a.scale;
	device const float *keys = qkv + a.d + h * 64 + 2 * lane;
	device const float *values = qkv + 2 * a.d + h * 64 + 2 * lane;
	float m = -INFINITY, l = 0;
	float2 acc = 0;
	for (uint j0 = w0; j0 < w1; j0 += AK) {
		float s[AK];
		float bm = -INFINITY;
		for (uint u = 0; u < AK; u++)
			s[u] = dot(q, *(device const float2 *)(keys + (ulong)min(j0 + u, w1 - 1) * stride));
		for (uint u = 0; u < AK; u++) {
			s[u] = j0 + u < w1 ? simd_sum(s[u]) : -INFINITY;
			bm = max(bm, s[u]);
		}
		float mn = max(m, bm), c = exp(m - mn);
		acc *= c;
		l *= c;
		for (uint u = 0; u < AK; u++) {
			float p = exp(s[u] - mn);
			acc += p * *(device const float2 *)(values + (ulong)min(j0 + u, w1 - 1) * stride);
			l += p;
		}
		m = mn;
	}
	*(device float2 *)(ctx + (ulong)r * a.d + h * 64 + 2 * lane) = acc / l;
}
