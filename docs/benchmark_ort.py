#!/usr/bin/env python3
"""Measure the pinned TinyMelNet graph with ONNX Runtime's CPU provider.

Each reported sample is the mean of a batch of session.run calls, including
Python call and result-allocation overhead. Conversion and feature extraction
are outside the timed region. Run from an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time


EXPECTED_SHA256 = "6b986a0440b30f533f0f7e473695939c347a2c95c5280eb2e090d0076b3dbbd1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--features", type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata/tone.mel.f32le",
    )
    parser.add_argument("--threads", nargs="+", type=positive, default=[1, 2, 4, 8])
    parser.add_argument("--warmups", type=positive, default=10)
    parser.add_argument("--batches", type=positive, default=3)
    parser.add_argument("--iterations", type=positive, default=200)
    parser.add_argument("--label", default="", help="hardware label, e.g. Apple M4 Max")
    parser.add_argument("--output", type=Path, help="also save the JSON report")
    args = parser.parse_args()

    digest = sha256(args.checkpoint)
    if digest != EXPECTED_SHA256:
        parser.error(f"checkpoint SHA-256 {digest} differs from {EXPECTED_SHA256}")

    import numpy as np
    import onnxruntime as ort

    features = np.fromfile(args.features, dtype="<f4")
    if features.size != 80 * 800 or not np.isfinite(features).all():
        parser.error("features must contain exactly 64,000 finite float32 values")
    features = np.ascontiguousarray(features.reshape(1, 80, 800), dtype=np.float32)
    inputs = {"mel": features}
    report = {
        "hardware": args.label,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "checkpointSha256": digest,
        "featuresSha256": sha256(args.features),
        "provider": "CPUExecutionProvider",
        "executionMode": "sequential",
        "graphOptimization": "all",
        "interOpThreads": 1,
        "warmups": args.warmups,
        "batches": args.batches,
        "iterationsPerBatch": args.iterations,
        "measurement": "mean milliseconds per session.run, including Python call overhead",
        "results": [],
    }
    for threads in args.threads:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(args.checkpoint), options, providers=["CPUExecutionProvider"]
        )
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(f"unexpected providers: {session.get_providers()}")
        for _ in range(args.warmups):
            session.run(None, inputs)
        means = []
        for _ in range(args.batches):
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                result = session.run(None, inputs)
            means.append((time.perf_counter_ns() - start) / args.iterations / 1e6)
        logit = float(result[0].reshape(-1)[0])
        if not math.isfinite(logit):
            raise RuntimeError(f"non-finite model output: {logit}")
        probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        report["results"].append({
            "intraOpThreads": threads,
            "batchMeansMs": means,
            "medianBatchMeanMs": statistics.median(means),
            "logit": logit,
            "probability": probability,
        })
        del session
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
