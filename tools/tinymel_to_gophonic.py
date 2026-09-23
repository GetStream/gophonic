#!/usr/bin/env python3
"""Convert the pinned TinyMelNet INT8 ONNX checkpoint and create CPU ORT oracles.

Bundle format (all integers and payloads are little-endian):

    8 bytes  magic ``GOTMEL1\0``
    u32      format version (1)
    u32      tensor count
    32 bytes SHA-256 of the source ONNX checkpoint
    repeated tensor records, sorted by name:
        u16      UTF-8 name byte count
        bytes    UTF-8 tensor name (ONNX value name, unchanged)
        u8       dtype tag: 1=float32, 2=uint8, 3=int64, 4=int32
        u8       rank
        u32[]    dimensions in ONNX order
        u32      element count (scalars have one element)
        bytes    contiguous row-major tensor contents in the tagged dtype

The bundle contains all 51 ONNX initializers and all 25 tensor-valued Constant
outputs. It preserves ONNX layouts and affine-quantization tensors; uint8
weights are not dequantized. The graph sequence and op attributes are emitted
to the oracle JSON when ``--oracle-dir`` is supplied. Runtime code implements
the graph; this utility deliberately does not simplify, fuse, or rewrite it.

Oracle inputs are a zero mel tensor, the repository's saved tone mel tensor,
and a deterministic 80x800 pattern. Expected raw logit and sigmoid probability
are saved once each in ``tinymel_oracle_logits.f32le``. The oracle JSON also
contains selected intermediate samples, full-tensor SHA-256 values, shapes,
quantization scales and zero points, and the ONNX node sequence. ORT is run
with CPUExecutionProvider only. The probability is sigmoid(raw logit), exactly
once; the checkpoint itself outputs logits.

The tone case also emits complete float32 little-endian files for the GRU input,
sequence output, and final hidden state, with the original ONNX shapes recorded
in the JSON manifest.

Requires Python packages numpy, onnx, and onnxruntime.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


MAGIC = b"GOTMEL1\0"
VERSION = 1
EXPECTED_ONNX_SHA256 = "6b986a0440b30f533f0f7e473695939c347a2c95c5280eb2e090d0076b3dbbd1"
MODEL_ID = "deveshu/hinglish-turn-detector/model_tinymel_int8.onnx"
INPUT_SHAPE = (1, 80, 800)
DTYPE_TAGS = {
    np.dtype("float32"): 1,
    np.dtype("uint8"): 2,
    np.dtype("int64"): 3,
    np.dtype("int32"): 4,
}
DTYPE_NAMES = {value: key.name for key, value in DTYPE_TAGS.items()}


# Outputs at the graph boundaries which are useful when locating a mismatch.
# The complete quantization trace is described separately below.
FLOAT_CHECKPOINTS = [
    "/stem/stem.0/Conv_output_0",
    "/stem/stem.2/Mul_1_output_0",
    "/stem/stem.3/depthwise/Conv_output_0",
    "/stem/stem.3/pointwise/Conv_output_0",
    "/stem/stem.3/act/Mul_1_output_0",
    "/stem/stem.4/depthwise/Conv_output_0",
    "/stem/stem.4/pointwise/Conv_output_0",
    "/stem/stem.4/act/Mul_1_output_0",
    "/stem/stem.5/depthwise/Conv_output_0",
    "/stem/stem.5/pointwise/Conv_output_0",
    "/stem/stem.5/act/Mul_1_output_0",
    "/gru/Transpose_output_0",
    "/gru/GRU_output_0",
    "/gru/GRU_output_1",
    "/gru/Transpose_2_output_0",
    "/pool/MatMul_output_0",
    "/pool/Softmax_output_0",
    "/pool/ReduceSum_output_0",
    "/head/head.0/LayerNormalization_output_0",
    "/head/head.1/Gemm_output_0",
    "/head/head.2/Mul_1_output_0",
    "/head/head.4/Gemm_output_0",
    "logit",
]

# Each entry links the explicit DynamicQuantizeLinear node outputs. Values are
# read from the graph itself, rather than re-derived from naming assumptions.
QUANTIZATION_GROUPS = [
    {
        "node": "mel_QuantizeLinear",
        "input": "mel",
        "quantized": "mel_quantized",
        "scale": "mel_scale",
        "zeroPoint": "mel_zero_point",
    },
    {
        "node": "/stem/stem.2/Mul_1_output_0_QuantizeLinear",
        "input": "/stem/stem.2/Mul_1_output_0",
        "quantized": "/stem/stem.2/Mul_1_output_0_quantized",
        "scale": "/stem/stem.2/Mul_1_output_0_scale",
        "zeroPoint": "/stem/stem.2/Mul_1_output_0_zero_point",
    },
    {
        "node": "/stem/stem.3/depthwise/Conv_output_0_QuantizeLinear",
        "input": "/stem/stem.3/depthwise/Conv_output_0",
        "quantized": "/stem/stem.3/depthwise/Conv_output_0_quantized",
        "scale": "/stem/stem.3/depthwise/Conv_output_0_scale",
        "zeroPoint": "/stem/stem.3/depthwise/Conv_output_0_zero_point",
    },
    {
        "node": "/stem/stem.3/act/Mul_1_output_0_QuantizeLinear",
        "input": "/stem/stem.3/act/Mul_1_output_0",
        "quantized": "/stem/stem.3/act/Mul_1_output_0_quantized",
        "scale": "/stem/stem.3/act/Mul_1_output_0_scale",
        "zeroPoint": "/stem/stem.3/act/Mul_1_output_0_zero_point",
    },
    {
        "node": "/stem/stem.4/depthwise/Conv_output_0_QuantizeLinear",
        "input": "/stem/stem.4/depthwise/Conv_output_0",
        "quantized": "/stem/stem.4/depthwise/Conv_output_0_quantized",
        "scale": "/stem/stem.4/depthwise/Conv_output_0_scale",
        "zeroPoint": "/stem/stem.4/depthwise/Conv_output_0_zero_point",
    },
    {
        "node": "/stem/stem.4/act/Mul_1_output_0_QuantizeLinear",
        "input": "/stem/stem.4/act/Mul_1_output_0",
        "quantized": "/stem/stem.4/act/Mul_1_output_0_quantized",
        "scale": "/stem/stem.4/act/Mul_1_output_0_scale",
        "zeroPoint": "/stem/stem.4/act/Mul_1_output_0_zero_point",
    },
    {
        "node": "/stem/stem.5/depthwise/Conv_output_0_QuantizeLinear",
        "input": "/stem/stem.5/depthwise/Conv_output_0",
        "quantized": "/stem/stem.5/depthwise/Conv_output_0_quantized",
        "scale": "/stem/stem.5/depthwise/Conv_output_0_scale",
        "zeroPoint": "/stem/stem.5/depthwise/Conv_output_0_zero_point",
    },
    {
        "node": "/gru/Transpose_2_output_0_QuantizeLinear",
        "input": "/gru/Transpose_2_output_0",
        "quantized": "/gru/Transpose_2_output_0_quantized",
        "scale": "/gru/Transpose_2_output_0_scale",
        "zeroPoint": "/gru/Transpose_2_output_0_zero_point",
    },
    {
        "node": "/head/head.0/LayerNormalization_output_0_QuantizeLinear",
        "input": "/head/head.0/LayerNormalization_output_0",
        "quantized": "/head/head.0/LayerNormalization_output_0_quantized",
        "scale": "/head/head.0/LayerNormalization_output_0_scale",
        "zeroPoint": "/head/head.0/LayerNormalization_output_0_zero_point",
    },
    {
        "node": "/head/head.2/Mul_1_output_0_QuantizeLinear",
        "input": "/head/head.2/Mul_1_output_0",
        "quantized": "/head/head.2/Mul_1_output_0_quantized",
        "scale": "/head/head.2/Mul_1_output_0_scale",
        "zeroPoint": "/head/head.2/Mul_1_output_0_zero_point",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> str:
    digest = sha256_file(path)
    if digest != EXPECTED_ONNX_SHA256:
        raise ValueError(
            f"unsupported ONNX checkpoint SHA-256: {digest} "
            f"(expected {EXPECTED_ONNX_SHA256})"
        )
    return digest


def convert_tensor(value: np.ndarray, source_name: str) -> tuple[np.ndarray, int]:
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder("=")
    if dtype not in DTYPE_TAGS:
        raise ValueError(f"{source_name} has unsupported dtype {array.dtype}")
    tag = DTYPE_TAGS[dtype]
    little_dtype = array.dtype.newbyteorder("<")
    array = array.astype(little_dtype, copy=False)
    # np.ascontiguousarray promotes rank-0 arrays to shape (1,); preserve the
    # scalar rank from ONNX while still normalizing non-contiguous tensors.
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array, tag


def collect_tensors(model: onnx.ModelProto) -> dict[str, tuple[np.ndarray, int]]:
    tensors: dict[str, tuple[np.ndarray, int]] = {}
    for initializer in model.graph.initializer:
        if initializer.name in tensors:
            raise ValueError(f"duplicate ONNX value {initializer.name!r}")
        array, tag = convert_tensor(numpy_helper.to_array(initializer), initializer.name)
        tensors[initializer.name] = (array, tag)

    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        if len(node.output) != 1:
            raise ValueError(f"Constant node {node.name!r} must have one output")
        value = next((attr.t for attr in node.attribute if attr.name == "value"), None)
        if value is None:
            raise ValueError(f"Constant node {node.name!r} is not tensor-valued")
        name = node.output[0]
        if name in tensors:
            raise ValueError(f"duplicate ONNX value {name!r}")
        array, tag = convert_tensor(numpy_helper.to_array(value), name)
        tensors[name] = (array, tag)

    if len(model.graph.initializer) != 51:
        raise ValueError(f"expected 51 initializers, found {len(model.graph.initializer)}")
    constant_count = sum(node.op_type == "Constant" for node in model.graph.node)
    if constant_count != 25:
        raise ValueError(f"expected 25 Constant nodes, found {constant_count}")
    if len(tensors) != 76:
        raise ValueError(f"expected 76 weight/constant tensors, found {len(tensors)}")
    return tensors


def write_bundle(path: Path, tensors: dict[str, tuple[np.ndarray, int]], digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as target:
            target.write(MAGIC)
            target.write(struct.pack("<II", VERSION, len(tensors)))
            target.write(bytes.fromhex(digest))
            for name in sorted(tensors):
                array, tag = tensors[name]
                encoded = name.encode("utf-8")
                shape = tuple(int(dimension) for dimension in array.shape)
                if len(encoded) == 0 or len(encoded) > 0xFFFF:
                    raise ValueError(f"invalid tensor name length for {name!r}")
                if len(shape) > 0xFF or any(d <= 0 or d > 0xFFFFFFFF for d in shape):
                    raise ValueError(f"unsupported tensor shape {shape} for {name}")
                count = int(array.size)
                if count > 0xFFFFFFFF:
                    raise ValueError(f"too many values for tensor {name}")
                target.write(struct.pack("<H", len(encoded)))
                target.write(encoded)
                target.write(struct.pack("<BB", tag, len(shape)))
                if shape:
                    target.write(struct.pack("<" + "I" * len(shape), *shape))
                target.write(struct.pack("<I", count))
                target.write(array.tobytes(order="C"))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def tensor_description(name: str, info_by_name: dict[str, onnx.ValueInfoProto]) -> dict[str, Any]:
    info = info_by_name.get(name)
    if info is None:
        raise ValueError(f"missing inferred ONNX value info for {name!r}")
    tensor_type = info.type.tensor_type
    dtype = TensorProto.DataType.Name(tensor_type.elem_type).lower()
    shape = []
    for dimension in tensor_type.shape.dim:
        shape.append(dimension.dim_value if dimension.dim_value else dimension.dim_param or None)
    return {"name": name, "dtype": dtype, "shape": shape}


def node_attribute_manifest(node: onnx.NodeProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in node.attribute:
        value = helper.get_attribute_value(attribute)
        if attribute.type == onnx.AttributeProto.TENSOR:
            array = numpy_helper.to_array(value)
            array, _ = convert_tensor(array, f"{node.name}:{attribute.name}")
            result[attribute.name] = {
                "dtype": DTYPE_NAMES[DTYPE_TAGS[array.dtype.newbyteorder("=")]].lower(),
                "shape": list(array.shape),
                "sha256le": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        else:
            result[attribute.name] = json_value(value)
    return result


def graph_manifest(model: onnx.ModelProto, info_by_name: dict[str, onnx.ValueInfoProto]) -> dict[str, Any]:
    nodes = []
    for index, node in enumerate(model.graph.node):
        nodes.append(
            {
                "index": index,
                "name": node.name,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": [tensor_description(name, info_by_name) for name in node.output if name],
                "attributes": node_attribute_manifest(node),
            }
        )
    quantization = []
    for group in QUANTIZATION_GROUPS:
        quantization.append(
            {
                **group,
                "scaleDtype": tensor_description(group["scale"], info_by_name)["dtype"],
                "zeroPointDtype": tensor_description(group["zeroPoint"], info_by_name)["dtype"],
                "quantizedDtype": tensor_description(group["quantized"], info_by_name)["dtype"],
                "quantizedShape": tensor_description(group["quantized"], info_by_name)["shape"],
                "rule": "round-to-nearest-even(x / scale) + zeroPoint, clamped to uint8 [0,255]",
            }
        )
    return {
        "nodes": nodes,
        "quantization": quantization,
        "architecture": {
            "input": {"name": "mel", "dtype": "float32", "shape": list(INPUT_SHAPE)},
            "output": {"name": "logit", "dtype": "float32", "shape": [1]},
            "stem": "1D ConvInteger (192 channels, kernel 5, stride 2, pad 2), exact GELU",
            "blocks": [
                {"name": "stem.3", "depthwise": "kernel 5, stride 2, pad 2, groups 192", "pointwise": "kernel 1, 192 channels", "activation": "exact GELU"},
                {"name": "stem.4", "depthwise": "kernel 5, stride 2, pad 2, groups 192", "pointwise": "kernel 1, 192 channels", "activation": "exact GELU"},
                {"name": "stem.5", "depthwise": "kernel 5, stride 1, pad 2, groups 192", "pointwise": "kernel 1, 192 channels", "activation": "exact GELU"},
            ],
            "gru": {"direction": "bidirectional", "hiddenSize": 128, "linearBeforeReset": 1, "gateOrder": "z,r,h", "inputShape": [100, 1, 192], "outputShape": [100, 2, 1, 128]},
            "pool": "learned query scalar scores, softmax over 100 steps, weighted sum to 256 features",
            "head": "LayerNormalization(256), INT8/UINT8 linear 256->256, exact GELU, INT8/UINT8 linear 256->1",
        },
    }


def fixture_pattern() -> np.ndarray:
    # Channel-major 80 x 800 values, fixed formulas and float32 rounding.
    channels = np.arange(80, dtype=np.float32)[:, None]
    frames = np.arange(800, dtype=np.float32)[None, :]
    values = -4.0 + 2.25 * np.sin(frames * np.float32(0.03125) + channels * np.float32(0.109375))
    values += 0.75 * np.cos(frames * np.float32(0.00390625) - channels * np.float32(0.0625))
    return np.ascontiguousarray(values[None, :, :], dtype=np.float32)


def load_mel(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    expected = math.prod(INPUT_SHAPE) * 4
    if len(raw) != expected:
        raise ValueError(f"{path} has {len(raw)} bytes; expected {expected} for float32 shape {INPUT_SHAPE}")
    return np.frombuffer(raw, dtype="<f4").reshape(INPUT_SHAPE).astype(np.float32, copy=False)


def little_endian_bytes(array: np.ndarray) -> bytes:
    array = np.asarray(array)
    dtype = array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array.astype(dtype, copy=False)).tobytes(order="C")


def sample_array(array: np.ndarray, limit: int = 12) -> dict[str, Any]:
    array = np.asarray(array)
    flat = array.reshape(-1)
    if flat.size <= limit:
        indices = list(range(flat.size))
    else:
        candidates = [0, 1, 2, 3, 11, flat.size // 16, flat.size // 4, flat.size // 2, 3 * flat.size // 4, flat.size - 4, flat.size - 2, flat.size - 1]
        indices = sorted(set(max(0, min(flat.size - 1, int(index))) for index in candidates))[:limit]
    values = [json_value(flat[index]) for index in indices]
    raw = little_endian_bytes(array)
    record: dict[str, Any] = {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "sha256le": hashlib.sha256(raw).hexdigest(),
        "sample": {"flatIndices": indices, "values": values},
    }
    if array.dtype.kind in "fi" and array.size:
        record["min"] = json_value(np.min(array))
        record["max"] = json_value(np.max(array))
        if array.dtype.kind == "f":
            record["mean"] = json_value(np.mean(array, dtype=np.float64))
    return record


def build_stage_model(model: onnx.ModelProto, info_by_name: dict[str, onnx.ValueInfoProto], extra_outputs: list[str]) -> onnx.ModelProto:
    stage_model = copy.deepcopy(model)
    output_names = {item.name for item in stage_model.graph.output}
    for name in extra_outputs:
        if name in output_names:
            continue
        info = info_by_name.get(name)
        if info is None:
            raise ValueError(f"missing shape/type metadata for ORT stage output {name!r}")
        stage_model.graph.output.append(copy.deepcopy(info))
        output_names.add(name)
    return stage_model


def run_oracles(
    checkpoint: Path,
    digest: str,
    oracle_dir: Path,
    tone_mel: Path,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("--oracle-dir requires onnxruntime") from exc

    oracle_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(checkpoint, load_external_data=True)
    inferred = shape_inference.infer_shapes(model)
    info_by_name: dict[str, onnx.ValueInfoProto] = {}
    for item in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info):
        info_by_name[item.name] = item

    dql_outputs = []
    for group in QUANTIZATION_GROUPS:
        dql_outputs.extend((group["quantized"], group["scale"], group["zeroPoint"]))
    stage_outputs = list(dict.fromkeys([*FLOAT_CHECKPOINTS, *dql_outputs]))
    stage_model = build_stage_model(inferred, info_by_name, stage_outputs)
    stage_model_bytes = stage_model.SerializeToString()

    cases: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    cases.append(("silence", np.zeros(INPUT_SHAPE, dtype=np.float32), {"kind": "zeros"}))
    tone = load_mel(tone_mel)
    cases.append(("tone", tone, {"kind": "file", "path": tone_mel.name, "sha256": sha256_file(tone_mel)}))
    pattern = fixture_pattern()
    pattern_path = oracle_dir / "tinymel_pattern.mel.f32le"
    pattern_path.write_bytes(little_endian_bytes(pattern))
    cases.append(("pattern", pattern, {"kind": "file", "path": pattern_path.name, "sha256": sha256_file(pattern_path)}))

    # ORT creates a small :memory:.ses side marker in its process working
    # directory on some builds. Keep all session work in an ephemeral cwd.
    old_cwd = Path.cwd()
    expected_outputs: list[float] = []
    case_results = []
    gru_fixtures: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="gophonic-tinymel-ort-") as scratch:
        try:
            os.chdir(scratch)
            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = min(8, os.cpu_count() or 1)
            original_session = ort.InferenceSession(
                checkpoint.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"]
            )
            stage_session = ort.InferenceSession(
                stage_model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
            )
            original_output = original_session.get_outputs()[0].name
            stage_names = [output.name for output in stage_session.get_outputs()]
            stage_index = {name: index for index, name in enumerate(stage_names)}
            for case_name, mel, input_description in cases:
                values = original_session.run([original_output], {"mel": mel})[0]
                logit = np.float32(np.asarray(values).reshape(-1)[0])
                probability = np.float32(1.0 / (1.0 + math.exp(-float(logit))))
                expected_outputs.extend((float(logit), float(probability)))

                stage_values = stage_session.run(stage_names, {"mel": mel})
                tensors: dict[str, Any] = {}
                for name in stage_outputs:
                    array = np.asarray(stage_values[stage_index[name]])
                    tensors[name] = sample_array(array)
                if case_name == "tone":
                    for tensor_name, filename in (
                        ("/gru/Transpose_output_0", "tinymel_tone_gru_input.f32le"),
                        ("/gru/GRU_output_0", "tinymel_tone_gru_output.f32le"),
                        ("/gru/GRU_output_1", "tinymel_tone_gru_final_hidden.f32le"),
                    ):
                        array = np.asarray(stage_values[stage_index[tensor_name]])
                        raw = little_endian_bytes(array)
                        fixture_path = oracle_dir / filename
                        fixture_path.write_bytes(raw)
                        gru_fixtures[tensor_name] = {
                            "file": filename,
                            "dtype": "float32",
                            "shape": list(array.shape),
                            "elementCount": int(array.size),
                            "sha256le": hashlib.sha256(raw).hexdigest(),
                            "layout": "row-major in ONNX dimension order",
                        }
                quantization = []
                for group in QUANTIZATION_GROUPS:
                    scale = float(np.asarray(stage_values[stage_index[group["scale"]]]).reshape(-1)[0])
                    zero_point = int(np.asarray(stage_values[stage_index[group["zeroPoint"]]]).reshape(-1)[0])
                    quantization.append(
                        {
                            "node": group["node"],
                            "input": group["input"],
                            "scale": scale,
                            "zeroPoint": zero_point,
                            "quantizedTensor": group["quantized"],
                        }
                    )
                case_results.append(
                    {
                        "name": case_name,
                        "input": {
                            **input_description,
                            "dtype": "float32",
                            "shape": list(INPUT_SHAPE),
                            "sha256le": hashlib.sha256(little_endian_bytes(mel)).hexdigest(),
                            "min": float(np.min(mel)),
                            "max": float(np.max(mel)),
                        },
                        "logit": float(logit),
                        "probability": float(probability),
                        "probabilityRule": "sigmoid(logit), applied once",
                        "quantization": quantization,
                        "tensors": tensors,
                    }
                )
            del stage_session
            del original_session
        finally:
            os.chdir(old_cwd)

    expected_path = oracle_dir / "tinymel_oracle_logits.f32le"
    expected_path.write_bytes(np.asarray(expected_outputs, dtype="<f4").tobytes(order="C"))
    manifest = {
        "format": "gophonic-tinymel-ort-oracle-v1",
        "model": MODEL_ID,
        "checkpointSha256": digest,
        "runtime": {
            "name": "ONNX Runtime",
            "version": ort.__version__,
            "provider": "CPUExecutionProvider",
            "executionMode": "sequential",
            "graphOptimization": "all",
            "intraOpThreads": min(8, os.cpu_count() or 1),
        },
        "bundle": {
            "magic": MAGIC.decode("ascii", errors="replace"),
            "version": VERSION,
            "initializerCount": 51,
            "constantCount": 25,
            "tensorCount": 76,
            "dtypeTags": {str(tag): name for tag, name in DTYPE_NAMES.items()},
        },
        "inputs": {"name": "mel", "dtype": "float32", "shape": list(INPUT_SHAPE), "layout": "[batch, mel_bin, frame]"},
        "output": {
            "name": "logit",
            "dtype": "float32",
            "shape": [1],
            "expectedFile": expected_path.name,
            "expectedFileLayout": "for each case in cases order: raw logit float32, sigmoid(logit) float32",
            "sha256": sha256_file(expected_path),
        },
        "architecture": graph_manifest(inferred, info_by_name),
        "floatCheckpoints": [tensor_description(name, info_by_name) for name in FLOAT_CHECKPOINTS],
        "gruFixtures": {
            "case": "tone",
            "inputMel": "tone.mel.f32le",
            "tensors": gru_fixtures,
        },
        "cases": case_results,
    }
    manifest_path = oracle_dir / "tinymel_oracle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False, allow_nan=False) + "\n")
    print(f"wrote {manifest_path} ({len(case_results)} cases, {len(stage_outputs)} stage tensors per case)")
    print(f"wrote {expected_path} ({len(expected_outputs)} float32 values: logit, probability per case)")
    print(f"wrote {oracle_dir / 'tinymel_pattern.mel.f32le'} ({math.prod(INPUT_SHAPE)} float32 values)")
    for fixture in gru_fixtures.values():
        print(f"wrote {oracle_dir / fixture['file']} ({fixture['elementCount']} float32 values)")


def validate_graph(model: onnx.ModelProto) -> None:
    input_names = [item.name for item in model.graph.input]
    output_names = [item.name for item in model.graph.output]
    if input_names != ["mel"]:
        raise ValueError(f"expected only input 'mel', got {input_names}")
    if output_names != ["logit"]:
        raise ValueError(f"expected only output 'logit', got {output_names}")
    input_shape = tuple(dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim)
    if input_shape != INPUT_SHAPE:
        raise ValueError(f"expected input shape {INPUT_SHAPE}, got {input_shape}")
    output_shape = tuple(dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim)
    if output_shape != (1,):
        raise ValueError(f"expected output shape (1,), got {output_shape}")
    if len(model.graph.node) != 133:
        raise ValueError(f"expected 133 graph nodes, found {len(model.graph.node)}")
    for group in QUANTIZATION_GROUPS:
        matches = [node for node in model.graph.node if node.name == group["node"] and node.op_type == "DynamicQuantizeLinear"]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one DynamicQuantizeLinear node named {group['node']!r}")
        if list(matches[0].output) != [group["quantized"], group["scale"], group["zeroPoint"]]:
            raise ValueError(f"unexpected outputs for quantizer {group['node']!r}: {list(matches[0].output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_model", type=Path, help="SHA-pinned model_tinymel_int8.onnx")
    parser.add_argument("--bundle", type=Path, help="write a GOTMEL1 v1 runtime weight/constant bundle")
    parser.add_argument("--oracle-dir", type=Path, help="write ORT outputs and stage fixtures under this directory")
    parser.add_argument(
        "--tone-mel",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "tone.mel.f32le",
        help="saved float32 1x80x800 mel tensor used as the tone fixture",
    )
    args = parser.parse_args()
    if args.bundle is None and args.oracle_dir is None:
        parser.error("specify --bundle, --oracle-dir, or both")

    checkpoint = args.onnx_model.resolve()
    digest = verify_checkpoint(checkpoint)
    model = onnx.load(checkpoint, load_external_data=True)
    validate_graph(model)
    tensors = collect_tensors(model)

    if args.bundle is not None:
        bundle = args.bundle.resolve()
        write_bundle(bundle, tensors, digest)
        print(f"wrote {bundle}: {len(tensors)} ONNX tensors, {bundle.stat().st_size:,} bytes")
    if args.oracle_dir is not None:
        tone_mel = args.tone_mel.resolve()
        run_oracles(checkpoint, digest, args.oracle_dir.resolve(), tone_mel)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"TinyMel conversion/oracle generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
