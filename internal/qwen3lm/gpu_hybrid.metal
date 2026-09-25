// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

// The Qwen3.5 family (Qwen3.6-35B-A3B): most layers mix tokens with a Gated
// DeltaNet, a linear attention whose memory is a matrix per head that
// decays and is corrected by the delta rule; every few layers attend, with
// wide heads, RoPE on their first dimensions, and a sigmoid gate on the
// output. Projections and experts use gpu.metal's kernels.
//
// A DeltaNet layer's input projection writes, per token, DNIN values: q, k,
// and v (the convolution's channels), then z (the output gate), b (the
// write strength), and a (the decay).

#ifndef HD
#define HD 256    // attention head width
#define HROT 64   // rotated dimensions of each attention head
#define DNK 16    // DeltaNet query and key heads
#define DNV 32    // DeltaNet value heads
#define DK 128    // query and key head width
#define DV 128    // value head width
#define DCONV 4   // causal convolution width
#endif
#define DNQKV (2 * DNK * DK + DNV * DV)
#define DNIN (DNQKV + DNV * DV + 2 * DNV)
#define HKVD (NKV * HD)                  // one attention row of keys or values
#define HQKVW (2 * NH * HD + 2 * HKVD)   // q and gate per head, then k and v

struct DnArgs {
	uint rows; // tokens in this pass
	float eps;
	uint params; // the layer's offset in the parameter buffer, in floats
};

// A DeltaNet layer's parameters, in floats from DnArgs.params: -exp(A_log)
// and dt_bias per value head, the output norm's weight, then the causal
// convolution's weights, DCONV per channel, oldest input first.
#define DN_NEGA 0
#define DN_DT DNV
#define DN_NORM (2 * DNV)
#define DN_CONV (2 * DNV + DV)
#define DN_PARAMS (DN_CONV + DNQKV * DCONV)

// dn_prep runs the causal convolution and SiLU over one head's channels of
// token row t, reading earlier inputs from the rows before it or, before
// row 0, from the layer's convolution state (the last DCONV-1 inputs, oldest
// first). Query and key heads are L2-normalized, queries also scaled by
// 1/sqrt(DK); each value head's simdgroup also writes the token's decay
// exp(g) and write strength beta. Threadgroup (head, t) is one simdgroup:
// heads 0..DNK-1 are queries, then DNK keys, then DNV values.
kernel void dn_prep(device const float *lin [[buffer(0)]], device const float *cs [[buffer(1)]],
		device const float *prm [[buffer(2)]], device float *dnx [[buffer(3)]],
		device float2 *gb [[buffer(4)]], constant DnArgs &a [[buffer(5)]],
		uint2 ht [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
	uint head = ht.x, t = ht.y;
	prm += a.params;
	uint c0 = head < 2 * DNK ? head * DK : 2 * DNK * DK + (head - 2 * DNK) * DV;
	float x[4];
	float ss = 0;
	for (uint i = 0; i < 4; i++) {
		uint c = c0 + lane + 32 * i;
		float acc = 0;
		for (uint j = 0; j < DCONV; j++) {
			int s = int(t) - int(DCONV - 1) + int(j);
			float in = s >= 0 ? lin[(ulong)s * DNIN + c] : cs[(ulong)(int(DCONV - 1) + s) * DNQKV + c];
			acc = fma(prm[DN_CONV + c * DCONV + j], in, acc);
		}
		x[i] = acc / (1 + exp(-acc));
		ss += x[i] * x[i];
	}
	float scale = 1;
	if (head < 2 * DNK) {
		scale = rsqrt(simd_sum(ss) + 1e-6f);
		if (head < DNK)
			scale *= rsqrt(float(DK));
	}
	for (uint i = 0; i < 4; i++)
		dnx[(ulong)t * DNQKV + c0 + lane + 32 * i] = x[i] * scale;
	if (head >= 2 * DNK && lane == 0) {
		uint h = head - 2 * DNK;
		device const float *ba = lin + (ulong)t * DNIN + DNQKV + DNV * DV;
		float b = ba[h], da = ba[DNV + h] + prm[DN_DT + h];
		float sp = da > 20 ? da : log(1 + exp(da));
		gb[t * DNV + h] = float2(exp(prm[DN_NEGA + h] * sp), 1 / (1 + exp(-b)));
	}
}

// dn_conv_state keeps the last DCONV-1 convolution inputs after a pass of
// a.rows tokens. One thread per channel.
kernel void dn_conv_state(device const float *lin [[buffer(0)]], device float *cs [[buffer(1)]],
		constant DnArgs &a [[buffer(5)]], uint c [[thread_position_in_grid]]) {
	for (uint i = 0; i < DCONV - 1; i++) {
		int s = int(a.rows) - int(DCONV - 1) + int(i);
		// Reads run ahead of writes: i + rows > i.
		cs[(ulong)i * DNQKV + c] = s >= 0 ? lin[(ulong)s * DNIN + c] : cs[(ulong)(i + a.rows) * DNQKV + c];
	}
}

// dn_scan runs the gated delta rule over a.rows tokens for value head h
// (its queries and keys are head h/(DNV/DNK)): with the state S (DK×DV) decayed
// by d = exp(g), the correction delta = beta·(v − d·Sᵀk) is written along k,
// S ← d·S + k·deltaᵀ, and the output is Sᵀq. The state stays in registers
// for the whole pass: four lanes share column j of S, a quarter of its rows
// each. Threadgroup h has 4·DV threads.
kernel void dn_scan(device const float *dnx [[buffer(0)]], device const float2 *gb [[buffer(1)]],
		device float *S [[buffer(2)]], device float *out [[buffer(3)]], constant DnArgs &a [[buffer(5)]],
		uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	constexpr uint P = DK / 4;
	uint j = tid / 4, p = tid % 4, kh = h / (DNV / DNK);
	device float *s = S + ((ulong)h * DK + p * P) * DV + j;
	float st[P];
	for (uint i = 0; i < P; i++)
		st[i] = s[i * DV];
	for (uint t = 0; t < a.rows; t++) {
		device const float *row = dnx + (ulong)t * DNQKV;
		device const float4 *q = (device const float4 *)(row + kh * DK + p * P);
		device const float4 *k = (device const float4 *)(row + DNK * DK + kh * DK + p * P);
		float v = row[2 * DNK * DK + h * DV + j];
		float2 db = gb[t * DNV + h];
		float mem = 0;
		for (uint i = 0; i < P / 4; i++) {
			float4 kk = k[i];
			mem += st[4 * i] * kk.x + st[4 * i + 1] * kk.y + st[4 * i + 2] * kk.z + st[4 * i + 3] * kk.w;
		}
		mem += simd_shuffle_xor(mem, 1);
		mem += simd_shuffle_xor(mem, 2);
		float delta = (v - db.x * mem) * db.y;
		float o = 0;
		for (uint i = 0; i < P / 4; i++) {
			float4 kk = k[i], qq = q[i];
			st[4 * i] = fma(st[4 * i], db.x, kk.x * delta);
			st[4 * i + 1] = fma(st[4 * i + 1], db.x, kk.y * delta);
			st[4 * i + 2] = fma(st[4 * i + 2], db.x, kk.z * delta);
			st[4 * i + 3] = fma(st[4 * i + 3], db.x, kk.w * delta);
			o += st[4 * i] * qq.x + st[4 * i + 1] * qq.y + st[4 * i + 2] * qq.z + st[4 * i + 3] * qq.w;
		}
		o += simd_shuffle_xor(o, 1);
		o += simd_shuffle_xor(o, 2);
		if (p == 0)
			out[(ulong)t * DNV * DV + h * DV + j] = o;
	}
	for (uint i = 0; i < P; i++)
		s[i * DV] = st[i];
}

// dn_norm normalizes value head h of row t's output (RMS over DV, times the
// norm's weight) and gates it by SiLU(z) into ctx. One simdgroup each.
kernel void dn_norm(device const float *o [[buffer(0)]], device const float *lin [[buffer(1)]],
		device const float *prm [[buffer(2)]], device float *ctx [[buffer(3)]], constant DnArgs &a [[buffer(5)]],
		uint2 ht [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
	uint h = ht.x, t = ht.y;
	device const float *src = o + (ulong)t * DNV * DV + h * DV;
	device const float *z = lin + (ulong)t * DNIN + DNQKV + h * DV;
	float x[DV / 32];
	float ss = 0;
	for (uint i = 0; i < DV / 32; i++) {
		x[i] = src[lane + 32 * i];
		ss += x[i] * x[i];
	}
	float inv = rsqrt(simd_sum(ss) / DV + a.eps);
	device float *dst = ctx + (ulong)t * DNV * DV + h * DV;
	for (uint i = 0; i < DV / 32; i++) {
		uint c = lane + 32 * i;
		float g = z[c];
		dst[c] = x[i] * inv * prm[a.params + DN_NORM + c] * (g / (1 + exp(-g)));
	}
}

// ---- Gated attention with HD-wide heads ----
//
// A lane holds dims lane + 32i of a head, i < HD/32. RoPE rotates dims d
// and d + HROT/2 for d < HROT/2: the lane's first two values when HROT is 64.

constant constexpr uint HPER = HD / 32;

inline void hNormRope(thread float *v, device const float *w, device const float *rope, uint pos, AttnArgs a, uint lane) {
	float ss = 0;
	for (uint i = 0; i < HPER; i++)
		ss += v[i] * v[i];
	float inv = rsqrt(simd_sum(ss) / HD + a.eps);
	for (uint i = 0; i < HPER; i++)
		v[i] *= inv * w[lane + 32 * i];
	for (uint d0 = 0; d0 < HROT / 2; d0 += 32) {
		uint d = d0 + lane, i0 = d0 / 32, i1 = (d0 + HROT / 2) / 32;
		float c = rope[pos * (HROT / 2) + d], s = rope[a.ropeSin + pos * (HROT / 2) + d];
		float x0 = v[i0], x1 = v[i1];
		v[i0] = x0 * c - x1 * s;
		v[i1] = x1 * c + x0 * s;
	}
}

inline void hLoad(thread float *v, device const float *src, uint lane) {
	for (uint i = 0; i < HPER; i++)
		v[i] = src[lane + 32 * i];
}

inline void hStore(device float *dst, thread const float *v, uint lane) {
	for (uint i = 0; i < HPER; i++)
		dst[lane + 32 * i] = v[i];
}

// hGateStore writes a head's attention output times the sigmoid of its gate.
inline void hGateStore(device float *dst, thread const float *v, float scale, device const float *gate, uint lane) {
	for (uint i = 0; i < HPER; i++) {
		float g = gate[lane + 32 * i];
		dst[lane + 32 * i] = v[i] * scale / (1 + exp(-g));
	}
}

// hqkRope prepares batch rows as qkRope does, for gated heads: queries are
// normalized, rotated, and scaled in place; keys go to cache row base+m,
// values are copied beside them.
kernel void hqkRope(device float *qkv [[buffer(0)]], device float *kc [[buffer(1)]],
		device float *vc [[buffer(2)]], device const float *qn [[buffer(3)]],
		device const float *kn [[buffer(4)]], device const float *rope [[buffer(5)]],
		device const uint2 *info [[buffer(6)]], constant AttnArgs &a [[buffer(7)]],
		uint2 hm [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
	uint h = hm.x, m = hm.y, slot = a.base + m, pos = info[m].x;
	device float *row = qkv + (ulong)m * HQKVW;
	float v[HPER];
	if (h < NH) {
		device float *q = row + h * 2 * HD;
		hLoad(v, q, lane);
		hNormRope(v, qn, rope, pos, a, lane);
		for (uint i = 0; i < HPER; i++)
			v[i] *= a.scale;
		hStore(q, v, lane);
	} else if (h < NH + NKV) {
		hLoad(v, row + 2 * NH * HD + (h - NH) * HD, lane);
		hNormRope(v, kn, rope, pos, a, lane);
		hStore(kc + (ulong)slot * HKVD + (h - NH) * HD, v, lane);
	} else {
		hLoad(v, row + 2 * NH * HD + HKVD + (h - NH - NKV) * HD, lane);
		hStore(vc + (ulong)slot * HKVD + (h - NH - NKV) * HD, v, lane);
	}
}

// hattendM attends batch row m to a read-only prefix and then its own
// sequence's rows, as attendM does, and gates the result. Threadgroup (g, m)
// holds the GROUP query heads of key/value head g as simdgroups.
kernel void hattendM(device const float *qkv [[buffer(0)]], device const float *kc [[buffer(1)]],
		device const float *vc [[buffer(2)]], device const uint2 *info [[buffer(3)]],
		device const float *pkc [[buffer(4)]], device const float *pvc [[buffer(5)]],
		device float *ctx [[buffer(6)]], constant AttnArgs &a [[buffer(7)]],
		uint2 gm [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
		uint lane [[thread_index_in_simdgroup]]) {
	uint g = gm.x, m = gm.y, head = g * GROUP + sg;
	device const float *row = qkv + (ulong)m * HQKVW;
	float q[HPER], acc[HPER];
	hLoad(q, row + head * 2 * HD, lane);
	for (uint i = 0; i < HPER; i++)
		acc[i] = 0;
	float mx = -INFINITY, l = 0;
	for (uint pass = 0; pass < 2; pass++) {
		device const float *ks = pass == 0 ? pkc : kc;
		device const float *vs = pass == 0 ? pvc : vc;
		uint j0 = pass == 0 ? 0 : info[m].y, j1 = pass == 0 ? a.prefixLen : a.base + m + 1;
		for (uint j = j0; j < j1; j++) {
			device const float *k = ks + (ulong)j * HKVD + g * HD;
			float s = 0;
			for (uint i = 0; i < HPER; i++)
				s += q[i] * k[lane + 32 * i];
			s = simd_sum(s);
			float mn = max(mx, s), c = exp(mx - mn), p = exp(s - mn);
			device const float *v = vs + (ulong)j * HKVD + g * HD;
			for (uint i = 0; i < HPER; i++)
				acc[i] = acc[i] * c + p * v[lane + 32 * i];
			l = l * c + p;
			mx = mn;
		}
	}
	hGateStore(ctx + (ulong)m * NH * HD + head * HD, acc, 1 / l, row + head * 2 * HD + HD, lane);
}

// hattend1 handles one new token at position a.pos, as attend1 does: the
// first simdgroup of threadgroup g appends key/value head g to the cache,
// then each query head of the group attends with HAS simdgroups taking
// every HAS-th key, and their partial states are merged and gated.
constant constexpr uint HAS = 2;

kernel void hattend1(device const float *qkv [[buffer(0)]], device float *kc [[buffer(1)]],
		device float *vc [[buffer(2)]], device const float *qn [[buffer(3)]],
		device const float *kn [[buffer(4)]], device const float *rope [[buffer(5)]],
		device float *ctx [[buffer(6)]], constant AttnArgs &a [[buffer(7)]],
		uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
		uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float partAcc[GROUP * HAS][HPER][32];
	threadgroup float2 partML[GROUP * HAS];
	float v[HPER];
	if (sg == 0) {
		hLoad(v, qkv + 2 * NH * HD + g * HD, lane);
		hNormRope(v, kn, rope, a.pos, a, lane);
		hStore(kc + (ulong)a.pos * HKVD + g * HD, v, lane);
		hLoad(v, qkv + 2 * NH * HD + HKVD + g * HD, lane);
		hStore(vc + (ulong)a.pos * HKVD + g * HD, v, lane);
	}
	threadgroup_barrier(mem_flags::mem_device);
	uint hq = sg % GROUP, split = sg / GROUP, head = g * GROUP + hq;
	float q[HPER], acc[HPER];
	hLoad(q, qkv + head * 2 * HD, lane);
	hNormRope(q, qn, rope, a.pos, a, lane);
	for (uint i = 0; i < HPER; i++) {
		q[i] *= a.scale;
		acc[i] = 0;
	}
	float mx = -INFINITY, l = 0;
	for (uint j = split; j <= a.pos; j += HAS) {
		device const float *k = kc + (ulong)j * HKVD + g * HD;
		float s = 0;
		for (uint i = 0; i < HPER; i++)
			s += q[i] * k[lane + 32 * i];
		s = simd_sum(s);
		float mn = max(mx, s), c = exp(mx - mn), p = exp(s - mn);
		device const float *vv = vc + (ulong)j * HKVD + g * HD;
		for (uint i = 0; i < HPER; i++)
			acc[i] = acc[i] * c + p * vv[lane + 32 * i];
		l = l * c + p;
		mx = mn;
	}
	for (uint i = 0; i < HPER; i++)
		partAcc[sg][i][lane] = acc[i];
	if (lane == 0)
		partML[sg] = float2(mx, l);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (split != 0)
		return;
	float top = -INFINITY;
	for (uint s = 0; s < HAS; s++)
		top = max(top, partML[s * GROUP + hq].x);
	float sum = 0;
	for (uint i = 0; i < HPER; i++)
		acc[i] = 0;
	for (uint s = 0; s < HAS; s++) {
		float2 ml = partML[s * GROUP + hq];
		float c = ml.x == -INFINITY ? 0 : exp(ml.x - top);
		for (uint i = 0; i < HPER; i++)
			acc[i] += partAcc[s * GROUP + hq][i][lane] * c;
		sum += ml.y * c;
	}
	hGateStore(ctx + head * HD, acc, 1 / sum, qkv + head * 2 * HD + HD, lane);
}
