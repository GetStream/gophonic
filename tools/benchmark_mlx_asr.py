#!/usr/bin/env python3
"""Warm MLX Audio reference for gophonic's Qwen3-ASR speech fixtures.

Run in an isolated environment with mlx-audio installed. This is optional
benchmark tooling, not a gophonic runtime dependency. The decoder-q8 option
uses MLX affine group-32 quantization; it is not gophonic's rotated Q8B format.
"""

import argparse
import importlib.metadata
import json
import platform
import resource
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1] / "testdata")
    parser.add_argument("--precision", choices=("bf16", "decoder-q8"), default="bf16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--implementation-revision", default="unspecified")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1:
        parser.error("repeats and warmups must be positive")
    if not args.model.is_dir():
        parser.error("model must be an existing local checkpoint directory")

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_audio.stt import load

    def emit(record):
        print(json.dumps(record, ensure_ascii=False), flush=True)

    start = time.perf_counter()
    model = load(str(args.model.resolve()), strict=True)
    quantized_layers = 0
    if args.precision == "decoder-q8":
        inner = model._model
        if inner.lm_head is None:
            # Preserve the BF16 embedding lookup and quantize a separate head,
            # as gophonic does for this tied-embedding checkpoint.
            head = nn.Linear(inner.config.text_config.hidden_size, inner.vocab_size, bias=False)
            head.weight = inner.model.embed_tokens.weight
            inner.lm_head = head
        nn.quantize(
            inner,
            group_size=32,
            bits=8,
            class_predicate=lambda path, module: isinstance(module, nn.Linear)
            and (path.startswith("model.layers.") or path == "lm_head"),
        )
        quantized_layers = sum(isinstance(module, nn.QuantizedLinear) for _, module in inner.named_modules())
        if not quantized_layers or not isinstance(inner.lm_head, nn.QuantizedLinear):
            raise RuntimeError("Decoder/head quantization failed; refusing a mislabeled benchmark")
        mx.eval(model.parameters())
    mx.synchronize()
    emit({
        "kind": "environment",
        "load_seconds": time.perf_counter() - start,
        "mlx": importlib.metadata.version("mlx"),
        "mlx_audio": importlib.metadata.version("mlx-audio"),
        "implementation_revision": args.implementation_revision,
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "device": mx.device_info(),
        "model": args.model.name,
        "precision": args.precision,
        "quantized_layers": quantized_layers,
        "embedding_quantized": False,
        "encoder_quantized": False,
        "quantization_note": "MLX affine group-32; differs from rotated Q8B" if quantized_layers else "Official checkpoint weights",
    })
    clips = {
        "jfk": np.fromfile(args.fixtures / "whisper_jfk.pcm.f32le", dtype="<f4"),
        "zh": np.fromfile(args.fixtures / "qwen3asr/zh.pcm.s16le", dtype="<i2").astype(np.float32) / 32768,
    }
    for name, pcm in clips.items():
        if not len(pcm) or not np.isfinite(pcm).all():
            raise ValueError(f"Invalid fixture {name}")
        peak = float(np.max(np.abs(pcm)))
        if peak > 1:
            pcm = pcm / peak  # Same peak normalization as gophonic.
        for _ in range(args.warmups):
            model.generate(pcm, max_tokens=256, temperature=0, verbose=False)
            mx.synchronize()
        mx.reset_peak_memory()
        samples, texts = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = model.generate(pcm, max_tokens=256, temperature=0, verbose=False)
            mx.synchronize()
            samples.append(time.perf_counter() - start)
            texts.append(result.text)
        if len(set(texts)) != 1:
            raise RuntimeError(f"Greedy transcript changed between repetitions for {name}: {texts}")
        emit({
            "kind": "measurement",
            "clip": name,
            "audio_seconds": len(pcm) / 16000,
            "seconds": samples,
            "median_seconds": statistics.median(samples),
            "text": texts[0],
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generation_tokens,
            "active_device_bytes": mx.get_active_memory(),
            "peak_device_bytes": mx.get_peak_memory(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })


if __name__ == "__main__":
    main()
