#!/usr/bin/env python3
"""Convert the pinned official OpenAI tiny.en.pt checkpoint to Go FP32 weights.

The converter reads only the known PyTorch ZIP tensor format and accepts only
the exact official SHA-256. Neither PyTorch nor Python is used at inference time.
"""

import argparse
import collections
import hashlib
import io
import pickle
import struct
import zipfile

import numpy as np


SOURCE_SHA256 = "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
DIMS = {
    "n_mels": 80, "n_vocab": 51864, "n_audio_ctx": 1500,
    "n_audio_state": 384, "n_audio_head": 6, "n_audio_layer": 4,
    "n_text_ctx": 448, "n_text_state": 384, "n_text_head": 6,
    "n_text_layer": 4,
}


class Tensor:
    def __init__(self, storage, offset, shape, strides, requires_grad, hooks):
        self.storage = storage
        self.offset = offset
        self.shape = tuple(shape)
        self.strides = tuple(strides)


class RestrictedTorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "HalfStorage"): "f2",
            ("torch", "FloatStorage"): "f4",
            ("torch._utils", "_rebuild_tensor_v2"): Tensor,
        }
        try:
            return allowed[(module, name)]
        except KeyError as exc:
            raise ValueError(f"unsupported pickle global {module}.{name}") from exc

    def persistent_load(self, ref):
        if not isinstance(ref, tuple) or len(ref) != 5 or ref[0] != "storage":
            raise ValueError("unsupported PyTorch storage reference")
        return ref


def expected():
    specs = {
        "encoder.conv1.weight": (384, 80, 3),
        "encoder.conv1.bias": (384,),
        "encoder.conv2.weight": (384, 384, 3),
        "encoder.conv2.bias": (384,),
        "encoder.positional_embedding": (1500, 384),
        "encoder.ln_post.weight": (384,),
        "encoder.ln_post.bias": (384,),
        "decoder.token_embedding.weight": (51864, 384),
        "decoder.positional_embedding": (448, 384),
        "decoder.ln.weight": (384,),
        "decoder.ln.bias": (384,),
    }
    for part in ("encoder", "decoder"):
        for layer in range(4):
            prefix = f"{part}.blocks.{layer}."
            for attention in ("attn", "cross_attn"):
                if part == "encoder" and attention == "cross_attn":
                    continue
                p = prefix + attention + "."
                for name in ("query", "key", "value", "out"):
                    specs[p + name + ".weight"] = (384, 384)
                    if name != "key":
                        specs[p + name + ".bias"] = (384,)
                for name in ("weight", "bias"):
                    specs[prefix + attention + "_ln." + name] = (384,)
            specs[prefix + "mlp.0.weight"] = (1536, 384)
            specs[prefix + "mlp.0.bias"] = (1536,)
            specs[prefix + "mlp.2.weight"] = (384, 1536)
            specs[prefix + "mlp.2.bias"] = (384,)
            for name in ("weight", "bias"):
                specs[prefix + "mlp_ln." + name] = (384,)
    return specs


def convert(source, output):
    digest = hashlib.sha256()
    with open(source, "rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SOURCE_SHA256:
        raise ValueError(f"unsupported checkpoint SHA-256: {digest.hexdigest()}")
    specs = expected()
    with zipfile.ZipFile(source) as archive:
        metadata = RestrictedTorchUnpickler(io.BytesIO(archive.read("archive/data.pkl"))).load()
        if metadata.get("dims") != DIMS:
            raise ValueError("unexpected tiny.en dimensions")
        weights = metadata["model_state_dict"]
        if set(weights) != set(specs):
            raise ValueError(f"tensor set differs: missing={set(specs)-set(weights)}, extra={set(weights)-set(specs)}")
        with open(output, "wb") as dst:
            dst.write(b"WHISPER1")
            dst.write(struct.pack("<II", 1, len(specs)))
            for name, shape in specs.items():
                tensor = weights[name]
                if not isinstance(tensor, Tensor) or tensor.shape != shape:
                    raise ValueError(f"unexpected shape for {name}")
                kind, dtype, key, device, count = tensor.storage
                if kind != "storage" or dtype not in ("f2", "f4") or device != "cpu":
                    raise ValueError(f"unsupported tensor storage for {name}")
                elements = int(np.prod(shape))
                if tensor.offset < 0 or tensor.offset + sum(
                    (dim - 1) * step for dim, step in zip(shape, tensor.strides)
                ) >= count:
                    raise ValueError(f"tensor {name} exceeds its storage")
                raw = archive.read(f"archive/data/{key}")
                values = np.frombuffer(raw, dtype="<f2" if dtype == "f2" else "<f4")
                if values.size != count:
                    raise ValueError(f"tensor {name} has truncated storage")
                values = np.ndarray(
                    shape, dtype=values.dtype, buffer=raw,
                    offset=tensor.offset * values.dtype.itemsize,
                    strides=tuple(step * values.dtype.itemsize for step in tensor.strides),
                )
                name_bytes = name.encode("ascii")
                dst.write(struct.pack("<H", len(name_bytes)))
                dst.write(name_bytes)
                dst.write(struct.pack("<B", len(shape)))
                dst.write(struct.pack("<" + "I" * len(shape), *shape))
                dst.write(struct.pack("<I", elements))
                dst.write(values.astype("<f4", copy=False).tobytes(order="C"))
    # The runtime hashes all tensor records after the 16-byte header.
    digest = hashlib.sha256()
    with open(output, "rb") as src:
        src.seek(16)
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    with open(output, "ab") as dst:
        dst.write(digest.digest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="official tiny.en.pt")
    parser.add_argument("output", help="output FP32 .gophonic bundle")
    args = parser.parse_args()
    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
