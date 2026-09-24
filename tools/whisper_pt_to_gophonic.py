#!/usr/bin/env python3
"""Convert a pinned official OpenAI English checkpoint to Go FP32 weights.

The converter reads only the known PyTorch ZIP tensor format and accepts only
the exact official SHA-256 of tiny.en, base.en, small.en, or medium.en.
tiny.en keeps the original version-1 bundle; larger models write version 2,
which records their dimensions inside the checksummed payload. Neither
PyTorch nor Python is used at inference time.
"""

import argparse
import collections
import hashlib
import io
import pickle
import struct
import zipfile

import numpy as np


def english_dims(state, heads, layers):
    return {
        "n_mels": 80, "n_vocab": 51864, "n_audio_ctx": 1500,
        "n_audio_state": state, "n_audio_head": heads, "n_audio_layer": layers,
        "n_text_ctx": 448, "n_text_state": state, "n_text_head": heads,
        "n_text_layer": layers,
    }


# Official checkpoints from openai/whisper __init__.py, keyed by SHA-256.
CHECKPOINTS = {
    "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03": ("tiny.en", english_dims(384, 6, 4)),
    "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead": ("base.en", english_dims(512, 8, 6)),
    "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872": ("small.en", english_dims(768, 12, 12)),
    "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f": ("medium.en", english_dims(1024, 16, 24)),
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


def expected(dims):
    a, t = dims["n_audio_state"], dims["n_text_state"]
    specs = {
        "encoder.conv1.weight": (a, 80, 3),
        "encoder.conv1.bias": (a,),
        "encoder.conv2.weight": (a, a, 3),
        "encoder.conv2.bias": (a,),
        "encoder.positional_embedding": (1500, a),
        "encoder.ln_post.weight": (a,),
        "encoder.ln_post.bias": (a,),
        "decoder.token_embedding.weight": (51864, t),
        "decoder.positional_embedding": (448, t),
        "decoder.ln.weight": (t,),
        "decoder.ln.bias": (t,),
    }
    for part in ("encoder", "decoder"):
        state = a if part == "encoder" else t
        layers = dims["n_audio_layer"] if part == "encoder" else dims["n_text_layer"]
        for layer in range(layers):
            prefix = f"{part}.blocks.{layer}."
            for attention in ("attn", "cross_attn"):
                if part == "encoder" and attention == "cross_attn":
                    continue
                p = prefix + attention + "."
                for name in ("query", "key", "value", "out"):
                    specs[p + name + ".weight"] = (state, state)
                    if name != "key":
                        specs[p + name + ".bias"] = (state,)
                for name in ("weight", "bias"):
                    specs[prefix + attention + "_ln." + name] = (state,)
            specs[prefix + "mlp.0.weight"] = (4 * state, state)
            specs[prefix + "mlp.0.bias"] = (4 * state,)
            specs[prefix + "mlp.2.weight"] = (state, 4 * state)
            specs[prefix + "mlp.2.bias"] = (state,)
            for name in ("weight", "bias"):
                specs[prefix + "mlp_ln." + name] = (state,)
    return specs


def convert(source, output):
    digest = hashlib.sha256()
    with open(source, "rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() not in CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint SHA-256: {digest.hexdigest()}")
    model, dims = CHECKPOINTS[digest.hexdigest()]
    specs = expected(dims)
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        root = names[0].split("/")[0]
        metadata = RestrictedTorchUnpickler(io.BytesIO(archive.read(f"{root}/data.pkl"))).load()
        if metadata.get("dims") != dims:
            raise ValueError(f"unexpected {model} dimensions: {metadata.get('dims')}")
        weights = metadata["model_state_dict"]
        if set(weights) != set(specs):
            raise ValueError(f"tensor set differs: missing={set(specs)-set(weights)}, extra={set(weights)-set(specs)}")
        with open(output, "wb") as dst:
            dst.write(b"WHISPER1")
            version = 1 if model == "tiny.en" else 2
            dst.write(struct.pack("<II", version, len(specs)))
            if version == 2:
                dst.write(struct.pack("<6I", dims["n_audio_state"], dims["n_audio_head"], dims["n_audio_layer"],
                                      dims["n_text_state"], dims["n_text_head"], dims["n_text_layer"]))
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
                raw = archive.read(f"{root}/data/{key}")
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
    parser.add_argument("checkpoint", help="official tiny.en.pt, base.en.pt, small.en.pt, or medium.en.pt")
    parser.add_argument("output", help="output FP32 .gophonic bundle")
    args = parser.parse_args()
    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
