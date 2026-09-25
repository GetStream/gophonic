#!/usr/bin/env python3
# Copyright 2026 The gophonic authors
# SPDX-License-Identifier: BSD-2-Clause
"""Reference forward pass of a Qwen3.5-family text decoder (Qwen3.6-35B-A3B),
in float32 NumPy, reading the official BF16 safetensors directly.

It follows transformers' Qwen3_5MoeTextModel: zero-centered RMSNorm (x times
1 + w), Gated DeltaNet layers (a causal depthwise convolution with SiLU over
q, k and v, L2-normalized q and k, the gated delta rule, and an RMSNorm
gated by SiLU(z)), and every fourth layer gated attention (q and a sigmoid
gate from one projection, q/k RMSNorm per head, RoPE on the first quarter of
each head); every layer's MLP is a mixture of experts, softmax top-k
renormalized, plus a shared expert scaled by its own sigmoid gate. Only the
experts the tokens choose are read.

    python3 qwen35_reference.py MODEL_DIR 760,6511,314,9338,369 out.f32

writes the last token's post-final-norm state (hidden float32 values) and
prints the head's top five tokens and logits.
"""
import json
import os
import struct
import sys

import numpy as np


class Checkpoint:
    def __init__(self, directory):
        self.dir = directory
        with open(os.path.join(directory, "model.safetensors.index.json")) as f:
            self.shard = json.load(f)["weight_map"]
        self.headers = {}

    def _meta(self, name):
        file = self.shard[name]
        if file not in self.headers:
            with open(os.path.join(self.dir, file), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.headers[file] = (json.loads(f.read(n)), 8 + n)
        header, base = self.headers[file]
        meta = header[name]
        assert meta["dtype"] == "BF16", (name, meta["dtype"])
        return file, base, meta

    def _read(self, file, offset, count):
        with open(os.path.join(self.dir, file), "rb") as f:
            f.seek(offset)
            raw = np.frombuffer(f.read(2 * count), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)

    def get(self, name):
        file, base, meta = self._meta(name)
        start, end = meta["data_offsets"]
        return self._read(file, base + start, (end - start) // 2).reshape(meta["shape"])

    def slab(self, name, index):
        """Returns tensor[index] of a tensor stacked along its first axis."""
        file, base, meta = self._meta(name)
        shape = meta["shape"]
        count = int(np.prod(shape[1:]))
        start = meta["data_offsets"][0] + 2 * index * count
        return self._read(file, base + start, count).reshape(shape[1:])


def rms(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0, x)


def l2norm(x, eps=1e-6):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def delta_net(ck, p, a, c):
    T = a.shape[0]
    nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    eps = c["rms_norm_eps"]
    qkv = a @ ck.get(p + "in_proj_qkv.weight").T  # [T, 2 nk dk + nv dv]
    w = ck.get(p + "conv1d.weight")[:, 0, :]  # [channels, kernel]
    kernel = w.shape[1]
    pad = np.concatenate([np.zeros((kernel - 1, qkv.shape[1]), np.float32), qkv])
    conv = np.zeros_like(qkv)
    for j in range(kernel):
        conv += pad[j : j + T] * w[:, j]
    conv = silu(conv)
    q = conv[:, : nk * dk].reshape(T, nk, dk)
    k = conv[:, nk * dk : 2 * nk * dk].reshape(T, nk, dk)
    v = conv[:, 2 * nk * dk :].reshape(T, nv, dv)
    z = (a @ ck.get(p + "in_proj_z.weight").T).reshape(T, nv, dv)
    beta = sigmoid(a @ ck.get(p + "in_proj_b.weight").T)  # [T, nv]
    g = -np.exp(ck.get(p + "A_log")) * softplus(a @ ck.get(p + "in_proj_a.weight").T + ck.get(p + "dt_bias"))
    rep = nv // nk
    q = l2norm(np.repeat(q, rep, axis=1)) / np.sqrt(dk)
    k = l2norm(np.repeat(k, rep, axis=1))
    S = np.zeros((nv, dk, dv), np.float32)
    out = np.zeros((T, nv, dv), np.float32)
    for t in range(T):
        S = S * np.exp(g[t])[:, None, None]
        mem = np.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - mem) * beta[t][:, None]
        S = S + np.einsum("hk,hv->hkv", k[t], delta)
        out[t] = np.einsum("hkv,hk->hv", S, q[t])
    out = rms(out, eps) * ck.get(p + "norm.weight") * silu(z)
    return out.reshape(T, nv * dv) @ ck.get(p + "out_proj.weight").T


def rope(x, positions, theta, dim):
    inv = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    ang = np.outer(positions, inv)
    cos = np.cos(np.concatenate([ang, ang], axis=-1)).astype(np.float32)[:, None, :]
    sin = np.sin(np.concatenate([ang, ang], axis=-1)).astype(np.float32)[:, None, :]
    r, keep = x[..., :dim], x[..., dim:]
    half = dim // 2
    rot = np.concatenate([-r[..., half:], r[..., :half]], axis=-1)
    return np.concatenate([r * cos + rot * sin, keep], axis=-1)


def attention(ck, p, a, c):
    T = a.shape[0]
    nh, nkv, hd, eps = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"]
    rp = c["rope_parameters"]
    dim = int(hd * rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1)))
    qg = (a @ ck.get(p + "q_proj.weight").T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)
    k = (a @ ck.get(p + "k_proj.weight").T).reshape(T, nkv, hd)
    v = (a @ ck.get(p + "v_proj.weight").T).reshape(T, nkv, hd)
    q = rms(q, eps) * (1 + ck.get(p + "q_norm.weight"))
    k = rms(k, eps) * (1 + ck.get(p + "k_norm.weight"))
    positions = np.arange(T)
    q, k = rope(q, positions, rp["rope_theta"], dim), rope(k, positions, rp["rope_theta"], dim)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    ctx = np.empty((T, nh, hd), np.float32)
    group = nh // nkv
    for h in range(nh):
        s = q[:, h] @ k[:, h // group].T / np.sqrt(hd) + mask
        s = np.exp(s - s.max(axis=-1, keepdims=True))
        ctx[:, h] = (s / s.sum(axis=-1, keepdims=True)) @ v[:, h // group]
    return (ctx.reshape(T, nh * hd) * sigmoid(gate)) @ ck.get(p + "o_proj.weight").T


def moe(ck, p, m, c, last):
    T = m.shape[0]
    topk, inter = c["num_experts_per_tok"], c["moe_intermediate_size"]
    logits = m @ ck.get(p + "gate.weight").T
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    chosen = np.argsort(-probs, axis=-1, kind="stable")[:, :topk]
    out = np.zeros_like(m)
    cache = {}
    for t in range(T):
        w = probs[t, chosen[t]]
        w = w / w.sum()
        for e, we in zip(chosen[t], w):
            if e not in cache:
                cache[e] = (ck.slab(p + "experts.gate_up_proj", e), ck.slab(p + "experts.down_proj", e))
            gu, d = cache[e]
            h = m[t] @ gu.T
            out[t] += we * ((silu(h[:inter]) * h[inter:]) @ d.T)
    shared = (silu(m @ ck.get(p + "shared_expert.gate_proj.weight").T) * (m @ ck.get(p + "shared_expert.up_proj.weight").T)) @ ck.get(
        p + "shared_expert.down_proj.weight"
    ).T
    out += sigmoid(m @ ck.get(p + "shared_expert_gate.weight").T) * shared
    last.append(chosen[-1].tolist())
    return out


def main():
    directory, ids, out = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3]
    with open(os.path.join(directory, "config.json")) as f:
        c = json.load(f)
    c = c.get("text_config", c)
    ck = Checkpoint(directory)
    pre = "model.language_model."
    eps = c["rms_norm_eps"]
    x = ck.get(pre + "embed_tokens.weight")[ids].astype(np.float32)
    for i, kind in enumerate(c["layer_types"]):
        p = f"{pre}layers.{i}."
        a = rms(x, eps) * (1 + ck.get(p + "input_layernorm.weight"))
        if kind == "linear_attention":
            x = x + delta_net(ck, p + "linear_attn.", a, c)
        else:
            x = x + attention(ck, p + "self_attn.", a, c)
        m = rms(x, eps) * (1 + ck.get(p + "post_attention_layernorm.weight"))
        last = []
        x = x + moe(ck, p + "mlp.", m, c, last)
        print(f"layer {i} ({kind}): experts of the last token {last[0]}", file=sys.stderr, flush=True)
    state = (rms(x[-1], eps) * (1 + ck.get(pre + "norm.weight"))).astype(np.float32)
    state.tofile(out)
    lg = ck.get("lm_head.weight") @ state
    top = np.argsort(-lg)[:5]
    print("top", [(int(t), float(lg[t])) for t in top])


if __name__ == "__main__":
    main()
