#!/usr/bin/env python3
# Copyright 2026 The gophonic authors
# SPDX-License-Identifier: BSD-2-Clause
"""Repackage a verified official English FP32 Go bundle as ggml for CPU comparison.

The ggml tiny.en template supplies only the standard Whisper mel filters and
tokenizer bytes, which every English checkpoint shares. Hyperparameters come
from the bundle, and every weight is replaced with the SHA-verified FP32
bundle's corresponding tensor. Pin the template's SHA-256 in the benchmark
record.
"""

import hashlib
import struct
import sys
from pathlib import Path


def u32(data, pos):
    return struct.unpack_from("<I", data, pos)[0], pos + 4


def read_ggml_prefix(path, dims):
    data = Path(path).read_bytes()
    pos = 0
    magic, pos = u32(data, pos)
    if magic != 0x67676D6C:
        raise ValueError("not a ggml Whisper model")
    hparams = []
    for _ in range(11):
        value, pos = u32(data, pos)
        hparams.append(value)
    n_mel, pos = u32(data, pos)
    n_freq, pos = u32(data, pos)
    pos += 4 * n_mel * n_freq
    n_vocab, pos = u32(data, pos)
    for _ in range(n_vocab):
        length, pos = u32(data, pos)
        pos += length
    if pos >= len(data) or n_mel != 80 or n_freq != 201 or hparams[0] != 51864:
        raise ValueError("unexpected ggml tiny.en header")
    # The final hparam is model-wide ftype; every tensor also has its own ftype.
    prefix = bytearray(data[:pos])
    # hparams: vocab, audio ctx/state/head/layer, text ctx/state/head/layer, mels, ftype.
    audio_state, audio_heads, audio_layers, text_state, text_heads, text_layers = dims
    for index, value in ((2, audio_state), (3, audio_heads), (4, audio_layers),
                         (6, text_state), (7, text_heads), (8, text_layers), (10, 0)):
        struct.pack_into("<I", prefix, 4 + index * 4, value)
    return prefix


def convert(go_bundle, ggml_template, output):
    raw = Path(go_bundle).read_bytes()
    if raw[:8] != b"WHISPER1":
        raise ValueError("not a Whisper Go bundle")
    version, count = struct.unpack_from("<II", raw, 8)
    if hashlib.sha256(raw[16:-32]).digest() != raw[-32:]:
        raise ValueError("Go bundle checksum mismatch")
    pos = 16
    if version == 1 and count == 167:
        dims = (384, 6, 4, 384, 6, 4)
    elif version == 2:
        dims = struct.unpack_from("<6I", raw, pos)
        pos += 24
    else:
        raise ValueError("unexpected Go bundle version or tensor count")
    with Path(output).open("wb") as dst:
        dst.write(read_ggml_prefix(ggml_template, dims))
        for _ in range(count):
            name_len = struct.unpack_from("<H", raw, pos)[0]
            pos += 2
            name = raw[pos:pos + name_len]
            pos += name_len
            ndim = raw[pos]
            pos += 1
            shape = struct.unpack_from("<" + "I" * ndim, raw, pos)
            pos += 4 * ndim
            elements, pos = u32(raw, pos)
            if elements != __import__("math").prod(shape):
                raise ValueError(f"bad tensor shape: {name!r}")
            values = memoryview(raw)[pos:pos + 4 * elements]
            pos += 4 * elements
            # The upstream converter presents convolution biases as [N, 1].
            if name in (b"encoder.conv1.bias", b"encoder.conv2.bias"):
                shape = (shape[0], 1)
            dst.write(struct.pack("<III", len(shape), name_len, 0))
            dst.write(struct.pack("<" + "I" * len(shape), *reversed(shape)))
            dst.write(name)
            dst.write(values)
    if pos != len(raw) - 32:
        raise ValueError("trailing Go bundle records")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: whisper_bundle_to_ggml_f32.py official.gophonic ggml-template.bin output-f32.bin")
    convert(*sys.argv[1:])
