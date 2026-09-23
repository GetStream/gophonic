#!/usr/bin/env python3
"""Convert the public Pipecat Smart Turn v3.2 FP32 ONNX weights to gofloor's bundle."""

import argparse
import hashlib
import os
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper


MAGIC = b"GOINFER1"
VERSION = 1
EXPECTED_ONNX_SHA256 = "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be"


def expected():
    tensors = [
        ("conv1.weight", (384, 80, 3)),
        ("conv1.bias", (384,)),
        ("conv2.weight", (384, 384, 3)),
        ("conv2.bias", (384,)),
        ("embed_positions.weight", (400, 384)),
    ]
    for layer in range(4):
        prefix = f"layers.{layer}."
        tensors.extend((prefix + name + ".weight", (384, 384)) for name in ("q", "k", "v", "out"))
        tensors.extend(
            [
                (prefix + "q.bias", (384,)),
                (prefix + "v.bias", (384,)),
                (prefix + "out.bias", (384,)),
                (prefix + "self_norm.weight", (384,)),
                (prefix + "self_norm.bias", (384,)),
                (prefix + "fc1.weight", (1536, 384)),
                (prefix + "fc1.bias", (1536,)),
                (prefix + "fc2.weight", (384, 1536)),
                (prefix + "fc2.bias", (384,)),
                (prefix + "final_norm.weight", (384,)),
                (prefix + "final_norm.bias", (384,)),
            ]
        )
    tensors.extend(
        [
            ("encoder_norm.weight", (384,)),
            ("encoder_norm.bias", (384,)),
            ("pool1.weight", (256, 384)),
            ("pool1.bias", (256,)),
            ("pool2.weight", (1, 256)),
            ("pool2.bias", (1,)),
            ("classifier1.weight", (256, 384)),
            ("classifier1.bias", (256,)),
            ("classifier_norm.weight", (256,)),
            ("classifier_norm.bias", (256,)),
            ("classifier2.weight", (64, 256)),
            ("classifier2.bias", (64,)),
            ("classifier3.weight", (1, 64)),
            ("classifier3.bias", (1,)),
        ]
    )
    return tensors


def mappings():
    result = {}

    def add(target, source, transpose=False):
        result[target] = (source, transpose)

    add("conv1.weight", "inner.encoder.conv1.weight")
    add("conv1.bias", "inner.encoder.conv1.bias")
    add("conv2.weight", "inner.encoder.conv2.weight")
    add("conv2.bias", "inner.encoder.conv2.bias")
    add("embed_positions.weight", "inner.encoder.embed_positions.weight")

    matrix_ids = {
        0: {"q": "val_17", "k": "val_25", "v": "val_32", "out": "val_75", "fc1": "val_79", "fc2": "val_88"},
        1: {"q": "val_92", "k": "val_100", "v": "val_107", "out": "val_148", "fc1": "val_152", "fc2": "val_161"},
        2: {"q": "val_165", "k": "val_173", "v": "val_180", "out": "val_221", "fc1": "val_225", "fc2": "val_234"},
        3: {"q": "val_238", "k": "val_246", "v": "val_253", "out": "val_294", "fc1": "val_298", "fc2": "val_307"},
    }
    for layer in range(4):
        prefix = f"layers.{layer}."
        upstream = f"inner.encoder.layers.{layer}."
        for name, source in matrix_ids[layer].items():
            add(prefix + name + ".weight", source, transpose=True)
        for name in ("q", "v", "out"):
            add(prefix + name + ".bias", upstream + f"self_attn.{name}_proj.bias")
        add(prefix + "self_norm.weight", upstream + "self_attn_layer_norm.weight")
        add(prefix + "self_norm.bias", upstream + "self_attn_layer_norm.bias")
        add(prefix + "fc1.bias", upstream + "fc1.bias")
        add(prefix + "fc2.bias", upstream + "fc2.bias")
        add(prefix + "final_norm.weight", upstream + "final_layer_norm.weight")
        add(prefix + "final_norm.bias", upstream + "final_layer_norm.bias")

    add("encoder_norm.weight", "inner.encoder.layer_norm.weight")
    add("encoder_norm.bias", "inner.encoder.layer_norm.bias")
    add("pool1.weight", "val_311", transpose=True)
    add("pool1.bias", "inner.pool_attention.0.bias")
    add("pool2.weight", "val_313", transpose=True)
    add("pool2.bias", "inner.pool_attention.2.bias")
    for target, source in (
        ("classifier1.weight", "inner.classifier.0.weight"),
        ("classifier1.bias", "inner.classifier.0.bias"),
        ("classifier_norm.weight", "inner.classifier.1.weight"),
        ("classifier_norm.bias", "inner.classifier.1.bias"),
        ("classifier2.weight", "inner.classifier.4.weight"),
        ("classifier2.bias", "inner.classifier.4.bias"),
        ("classifier3.weight", "inner.classifier.6.weight"),
        ("classifier3.bias", "inner.classifier.6.bias"),
    ):
        add(target, source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_model", help="smart-turn-v3.2-gpu.onnx from pipecat-ai/smart-turn-v3")
    parser.add_argument("output", help="output .gofloor weight bundle")
    args = parser.parse_args()

    digest = hashlib.sha256()
    with open(args.onnx_model, "rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_ONNX_SHA256:
        raise SystemExit(
            "unsupported ONNX checkpoint SHA-256: "
            f"{digest.hexdigest()} (expected {EXPECTED_ONNX_SHA256})"
        )

    model = onnx.load(args.onnx_model, load_external_data=True)
    graph = model.graph
    input_shapes = {
        value.name: tuple(dim.dim_value if dim.dim_value else dim.dim_param for dim in value.type.tensor_type.shape.dim)
        for value in graph.input
    }
    if "input_features" not in input_shapes or input_shapes["input_features"][-2:] != (80, 800):
        raise SystemExit(f"unsupported input shape: {input_shapes.get('input_features')}")
    if len(graph.output) != 1 or graph.output[0].name != "logits":
        raise SystemExit("expected the single Smart Turn probability output named 'logits'")

    initializers = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    name_map = mappings()
    specs = expected()
    if len(name_map) != len(specs):
        raise SystemExit(f"converter table has {len(name_map)} tensors but the bundle requires {len(specs)}")

    packed = []
    for target, shape in specs:
        if target not in name_map:
            raise SystemExit(f"no ONNX mapping for {target}")
        source, transpose = name_map[target]
        if source not in initializers:
            raise SystemExit(f"ONNX initializer {source!r} for {target} is missing")
        array = np.asarray(initializers[source])
        if array.dtype != np.float32:
            raise SystemExit(f"{source} has dtype {array.dtype}, expected float32")
        if transpose:
            if array.ndim != 2:
                raise SystemExit(f"cannot transpose rank-{array.ndim} initializer {source}")
            array = array.T
        if tuple(array.shape) != shape:
            raise SystemExit(f"{target} has shape {tuple(array.shape)}, expected {shape} (from {source})")
        packed.append((target, array.astype("<f4", copy=False).tobytes(order="C")))

    tmp = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(tmp, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<II", VERSION, len(packed)))
        for (name, shape), (_, data) in zip(specs, packed, strict=True):
            encoded = name.encode("utf-8")
            out.write(struct.pack("<H", len(encoded)))
            out.write(encoded)
            out.write(struct.pack("<B", len(shape)))
            out.write(struct.pack("<" + "I" * len(shape), *shape))
            out.write(struct.pack("<I", len(data) // 4))
            out.write(data)
    os.replace(tmp, args.output)
    print(f"wrote {args.output}: {len(packed)} tensors, {os.path.getsize(args.output):,} bytes")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
