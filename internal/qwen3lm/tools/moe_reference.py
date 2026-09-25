#!/usr/bin/env python3
# Copyright 2026 The gophonic authors
# SPDX-License-Identifier: BSD-2-Clause
"""Reference forward pass of a Qwen3-MoE checkpoint (Qwen3-30B-A3B), in
float32 NumPy, reading the official BF16 safetensors directly.

It follows transformers' Qwen3MoeForCausalLM: RMSNorm, q/k RMSNorm per head,
RoPE, grouped-query causal attention, and a mixture of experts whose router
softmax is renormalized over the top k. Only the experts the tokens choose
are read, so a few tokens take minutes, not the whole checkpoint.

    python3 moe_reference.py MODEL_DIR 785,6722,315,9625,374 out.f32

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
        index = os.path.join(directory, "model.safetensors.index.json")
        with open(index) as f:
            self.shard = json.load(f)["weight_map"]
        self.headers = {}

    def _header(self, file):
        if file not in self.headers:
            with open(os.path.join(self.dir, file), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.headers[file] = (json.loads(f.read(n)), 8 + n)
        return self.headers[file]

    def get(self, name):
        file = self.shard[name]
        header, base = self._header(file)
        meta = header[name]
        assert meta["dtype"] == "BF16", (name, meta["dtype"])
        start, end = meta["data_offsets"]
        with open(os.path.join(self.dir, file), "rb") as f:
            f.seek(base + start)
            raw = np.frombuffer(f.read(end - start), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, positions, theta):
    # x: [T, heads, d]; rotate_half on the two halves.
    d = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
    ang = np.outer(positions, inv)  # [T, d/2]
    cos = np.cos(np.concatenate([ang, ang], axis=-1)).astype(np.float32)[:, None, :]
    sin = np.sin(np.concatenate([ang, ang], axis=-1)).astype(np.float32)[:, None, :]
    half = d // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


def main():
    directory, ids, out = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3]
    with open(os.path.join(directory, "config.json")) as f:
        c = json.load(f)
    ck = Checkpoint(directory)
    h, nh, nkv, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps, theta, topk = c["rms_norm_eps"], c["rope_theta"], c["num_experts_per_tok"]
    T = len(ids)
    embed = ck.get("model.embed_tokens.weight")
    x = embed[ids].astype(np.float32)
    del embed
    positions = np.arange(T)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        a = rms_norm(x, ck.get(p + "input_layernorm.weight"), eps)
        q = (a @ ck.get(p + "self_attn.q_proj.weight").T).reshape(T, nh, hd)
        k = (a @ ck.get(p + "self_attn.k_proj.weight").T).reshape(T, nkv, hd)
        v = (a @ ck.get(p + "self_attn.v_proj.weight").T).reshape(T, nkv, hd)
        q = rope(rms_norm(q, ck.get(p + "self_attn.q_norm.weight"), eps), positions, theta)
        k = rope(rms_norm(k, ck.get(p + "self_attn.k_norm.weight"), eps), positions, theta)
        group = nh // nkv
        ctx = np.empty((T, nh, hd), dtype=np.float32)
        for head in range(nh):
            kh, vh = k[:, head // group], v[:, head // group]
            s = q[:, head] @ kh.T / np.sqrt(hd) + mask
            s = np.exp(s - s.max(axis=-1, keepdims=True))
            ctx[:, head] = (s / s.sum(axis=-1, keepdims=True)) @ vh
        x = x + ctx.reshape(T, nh * hd) @ ck.get(p + "self_attn.o_proj.weight").T
        m = rms_norm(x, ck.get(p + "post_attention_layernorm.weight"), eps)
        logits = m @ ck.get(p + "mlp.gate.weight").T  # [T, E]
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        out_moe = np.zeros_like(x)
        chosen = np.argsort(-probs, axis=-1, kind="stable")[:, :topk]
        cache = {}
        for t in range(T):
            w = probs[t, chosen[t]]
            w = w / w.sum()
            for e, we in zip(chosen[t], w):
                if e not in cache:
                    q_ = f"{p}mlp.experts.{e}."
                    cache[e] = (ck.get(q_ + "gate_proj.weight"), ck.get(q_ + "up_proj.weight"), ck.get(q_ + "down_proj.weight"))
                g, u, d = cache[e]
                gg = m[t] @ g.T
                act = gg / (1 + np.exp(-gg)) * (m[t] @ u.T)
                out_moe[t] += we * (act @ d.T)
        x = x + out_moe
        print(f"layer {i}: experts of the last token {chosen[-1].tolist()}", file=sys.stderr, flush=True)
    last = rms_norm(x[-1], ck.get("model.norm.weight"), eps).astype(np.float32)
    last.tofile(out)
    head = ck.get("lm_head.weight")
    lg = head @ last
    top = np.argsort(-lg)[:5]
    print("top", [(int(t), float(lg[t])) for t in top])


if __name__ == "__main__":
    main()
