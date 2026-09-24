#!/usr/bin/env python3
"""Convert a trusted CLM PyTorch head checkpoint to a pure-Go .gclm bundle.

This converter is offline: it never downloads code or weights. It uses
torch.load(weights_only=True), so the input must be a state-dict checkpoint
made from tensors and ordinary Python metadata (PyTorch 2.6+).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
import torch


MAGIC = b"GCLMCPU1"
VERSION = 1
DEFAULT_ENCODER_DIM = 4096
DEFAULT_PROJECTION_DIM = 512
MAX_DEPTH = 64
MAX_WIDTH = 16384
MAX_PROJECTION = 16384
MAX_PARAMETERS = 250_000_000
REFERENCE_REPO = "Contrastive-LM/CLM-v0.1-8B"
REFERENCE_REVISION = "87655cb835bd76fd66c2da78e1e3709f7fa11a94"
REFERENCE_FILE = "CLM_v0.1-8B.pt"


class ConversionError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MISSING = object()


def exact_int(mapping: dict, key: str, default=_MISSING, *, label: str) -> int:
    if key not in mapping:
        if default is _MISSING:
            raise ConversionError(f"{label} is required")
        return default
    value = mapping[key]
    # Reject bool as well: bool is an int subclass in Python.
    if type(value) is not int:
        raise ConversionError(f"{label} must be an integer, got {type(value).__name__}")
    return value


def exact_bool(mapping: dict, key: str, default: bool, *, label: str) -> bool:
    if key not in mapping:
        return default
    value = mapping[key]
    if type(value) is not bool:
        raise ConversionError(f"{label} must be a boolean, got {type(value).__name__}")
    return value


def exact_str(mapping: dict, key: str, default: str, *, label: str) -> str:
    if key not in mapping:
        return default
    value = mapping[key]
    if type(value) is not str:
        raise ConversionError(f"{label} must be a string, got {type(value).__name__}")
    return value


def config_from_checkpoint(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict):
        raise ConversionError("checkpoint root must be a dictionary")
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, dict):
        raise ConversionError("checkpoint must contain a cfg dictionary")
    projection_source = checkpoint if "projection_dim" in checkpoint else cfg
    projection_label = "checkpoint.projection_dim" if projection_source is checkpoint else "cfg.projection_dim"
    config = {
        "encoder_dim": exact_int(cfg, "hidden_size", DEFAULT_ENCODER_DIM, label="cfg.hidden_size"),
        "width": exact_int(cfg, "width", label="cfg.width"),
        "depth": exact_int(cfg, "depth", label="cfg.depth"),
        "projection_dim": exact_int(projection_source, "projection_dim", DEFAULT_PROJECTION_DIM, label=projection_label),
        "activation": exact_str(cfg, "activation", "gelu", label="cfg.activation"),
        "layernorm": exact_bool(cfg, "layernorm", False, label="cfg.layernorm"),
        "residual": exact_bool(cfg, "residual", False, label="cfg.residual"),
    }
    if not (0 < config["encoder_dim"] <= MAX_WIDTH):
        raise ConversionError(f"invalid encoder_dim {config['encoder_dim']}")
    if not (0 < config["width"] <= MAX_WIDTH):
        raise ConversionError(f"invalid width {config['width']}")
    if not (2 <= config["depth"] <= MAX_DEPTH):
        raise ConversionError(f"invalid depth {config['depth']}")
    if not (0 < config["projection_dim"] <= MAX_PROJECTION):
        raise ConversionError(f"invalid projection_dim {config['projection_dim']}")
    if config["activation"] not in {"gelu", "relu", "silu"}:
        raise ConversionError(f"unsupported activation {config['activation']!r}")
    count = 2 * (
        config["width"] * config["encoder_dim"] + config["width"]
        + (config["depth"] - 2) * (config["width"] * config["width"] + config["width"]
                                    + (2 * config["width"] if config["layernorm"] else 0))
        + config["projection_dim"] * config["width"] + config["projection_dim"]
    )
    if count > MAX_PARAMETERS:
        raise ConversionError(f"parameter count {count} exceeds limit {MAX_PARAMETERS}")
    return config


def tensor_specs(config: dict) -> list[tuple[str, tuple[int, ...], str]]:
    specs: list[tuple[str, tuple[int, ...], str]] = []
    for head in ("state_head", "action_head"):
        specs.append((f"{head}.inp.weight", (config["width"], config["encoder_dim"]), "inp.weight"))
        specs.append((f"{head}.inp.bias", (config["width"],), "inp.bias"))
        for i in range(config["depth"] - 2):
            specs.append((f"{head}.hidden.{i}.weight", (config["width"], config["width"]), f"hidden.{i}.weight"))
            specs.append((f"{head}.hidden.{i}.bias", (config["width"],), f"hidden.{i}.bias"))
            if config["layernorm"]:
                specs.append((f"{head}.hidden.{i}.norm.weight", (config["width"],), f"norms.{i}.weight"))
                specs.append((f"{head}.hidden.{i}.norm.bias", (config["width"],), f"norms.{i}.bias"))
        specs.append((f"{head}.out.weight", (config["projection_dim"], config["width"]), "out.weight"))
        specs.append((f"{head}.out.bias", (config["projection_dim"],), "out.bias"))
    return specs


def to_f32_bytes(tensor: torch.Tensor, name: str, shape: tuple[int, ...]) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise ConversionError(f"{name}: expected a tensor")
    if tuple(tensor.shape) != shape:
        raise ConversionError(f"{name}: got shape {tuple(tensor.shape)}, want {shape}")
    if tensor.is_complex() or tensor.is_quantized:
        raise ConversionError(f"{name}: complex or quantized tensors are unsupported")
    data = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    data = np.asarray(data, dtype="<f4", order="C")
    if not np.isfinite(data).all():
        raise ConversionError(f"{name}: tensor contains a non-finite value")
    return data.tobytes(order="C")


def convert(args: argparse.Namespace) -> None:
    source = Path(args.checkpoint).resolve()
    destination = Path(args.output).resolve()
    if source == destination:
        raise ConversionError("input checkpoint and output bundle must be different files")
    if not source.is_file():
        raise ConversionError(f"checkpoint does not exist: {source}")
    source_sha = sha256_file(source)
    if args.expected_sha256:
        expected = args.expected_sha256.lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ConversionError("--expected-sha256 must be 64 hexadecimal characters")
        if source_sha != expected:
            raise ConversionError(f"checkpoint SHA-256 mismatch: got {source_sha}, want {expected}")

    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConversionError(f"safe torch.load(weights_only=True) failed: {exc}") from exc
    config = config_from_checkpoint(checkpoint)
    if "state_head" not in checkpoint or "action_head" not in checkpoint:
        raise ConversionError("checkpoint must contain state_head and action_head state dictionaries")

    scale_tensor = checkpoint.get("logit_scale")
    if isinstance(scale_tensor, torch.Tensor):
        if scale_tensor.numel() != 1:
            raise ConversionError("logit_scale must contain one value")
        logit_scale = float(scale_tensor.detach().to(device="cpu", dtype=torch.float32).item())
    else:
        try:
            logit_scale = float(scale_tensor)
        except (TypeError, ValueError) as exc:
            raise ConversionError("logit_scale must be a scalar") from exc
        logit_scale = float(torch.tensor(logit_scale, dtype=torch.float32).item())
    if not math.isfinite(logit_scale):
        raise ConversionError("logit_scale must be finite")
    scale = float(torch.exp(torch.tensor(logit_scale, dtype=torch.float32)).clamp(max=100.0).item())
    if not math.isfinite(scale) or scale < 0 or scale > 100:
        raise ConversionError(f"invalid exp(logit_scale) value {scale}")

    payloads: list[bytes] = []
    tensor_records = []
    for head_name in ("state_head", "action_head"):
        state = checkpoint[head_name]
        if not isinstance(state, dict):
            raise ConversionError(f"{head_name} must be a state dictionary")
        # Validate this head independently against the architecture's registered keys.
        head_specs = [spec for spec in tensor_specs(config) if spec[0].startswith(head_name + ".")]
        expected_keys = {key for _, _, key in head_specs}
        actual_keys = set(state)
        missing, extra = sorted(expected_keys - actual_keys), sorted(actual_keys - expected_keys)
        if missing or extra:
            raise ConversionError(f"{head_name} keys differ: missing={missing}, unexpected={extra}")
        for name, shape, key in head_specs:
            raw = to_f32_bytes(state[key], name, shape)
            payloads.append(raw)
            tensor_records.append({"name": name, "shape": list(shape), "sha256": hashlib.sha256(raw).hexdigest()})

    manifest = {
        "format": "gophonic-clm",
        "version": VERSION,
        "config": config,
        "logit_scale": logit_scale,
        "scale": scale,
        "provenance": {
            "source_repo": args.source_repo,
            "source_revision": args.source_revision,
            "source_file": args.source_file or source.name,
            "source_sha256": source_sha,
        },
        "tensors": tensor_records,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(manifest_bytes) > (1 << 20):
        raise ConversionError("manifest is unexpectedly large")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(MAGIC)
            out.write(struct.pack("<II", VERSION, len(manifest_bytes)))
            out.write(manifest_bytes)
            for payload in payloads:
                out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_name, destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    print(f"wrote {destination} ({destination.stat().st_size} bytes)")
    print(f"source sha256: {source_sha}")
    print(f"projection scale: {scale:g}; encoder={config['encoder_dim']} width={config['width']} depth={config['depth']} projection={config['projection_dim']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="local CLM torch.save checkpoint (.pt)")
    parser.add_argument("output", help="output pure-Go bundle (.gclm)")
    parser.add_argument("--expected-sha256", default="", help="fail unless the local checkpoint has this SHA-256")
    parser.add_argument("--source-repo", default=REFERENCE_REPO, help="checkpoint repository recorded as provenance")
    parser.add_argument("--source-revision", default=REFERENCE_REVISION, help="checkpoint revision recorded as provenance")
    parser.add_argument("--source-file", default=REFERENCE_FILE, help="checkpoint filename recorded as provenance")
    args = parser.parse_args()
    try:
        convert(args)
    except (ConversionError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
