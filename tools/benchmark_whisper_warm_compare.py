#!/usr/bin/env python3
"""Alternate matched in-process Go and CPU-only whisper.cpp window runs.

Each helper loads its model, runs five untimed warm calls, then prints one
nanosecond sample per call. Run one helper process at a time to avoid contention.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(binary, model, pcm, workers, calls, cpp=False, expected=None):
    command = [str(binary.resolve()), str(model.resolve()), str(pcm.resolve()),
               str(workers), str(calls)]
    if expected:
        command.append(expected)
    env = os.environ.copy()
    if cpp:
        # Apple's BLAS thread budget is separate from whisper.cpp n_threads.
        env["VECLIB_MAXIMUM_THREADS"] = str(workers)
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    samples = [int(line) for line in result.stdout.splitlines()]
    if len(samples) != calls or any(sample <= 0 for sample in samples):
        raise RuntimeError(f"unexpected timing output from {binary}: {result.stdout}")
    return samples


def ratio_interval(go_blocks, cpp_blocks):
    # Resample entire blocks to preserve call-to-call correlation and pairing.
    rng = random.Random(0)
    n = len(go_blocks)
    ratios = []
    for _ in range(10000):
        indices = [rng.randrange(n) for _ in range(n)]
        go = [value for i in indices for value in go_blocks[i]]
        cpp = [value for i in indices for value in cpp_blocks[i]]
        ratios.append(statistics.median(go) / statistics.median(cpp))
    ratios.sort()
    return [ratios[250], ratios[9749]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("go_binary", type=Path)
    parser.add_argument("go_model", type=Path)
    parser.add_argument("cpp_binary", type=Path)
    parser.add_argument("cpp_model", type=Path)
    parser.add_argument("pcm", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--expected", help="expected transcript when it differs from tiny.en's")
    args = parser.parse_args()
    if min(args.workers, args.blocks, args.calls) < 1:
        parser.error("workers, blocks, and calls must be positive")

    go_blocks = []
    cpp_blocks = []
    for block in range(args.blocks):
        order = ("go", "cpp") if block % 2 == 0 else ("cpp", "go")
        for runtime in order:
            if runtime == "go":
                go_blocks.append(run(args.go_binary, args.go_model, args.pcm,
                                     args.workers, args.calls, expected=args.expected))
            else:
                cpp_blocks.append(run(args.cpp_binary, args.cpp_model, args.pcm,
                                      args.workers, args.calls, cpp=True, expected=args.expected))

    go = [value for block in go_blocks for value in block]
    cpp = [value for block in cpp_blocks for value in block]
    report = {
        "boundary": "reused 30-second PCM-to-text window, official FP32 weights",
        "expected_transcript": args.expected or "tiny.en JFK oracle",
        "workers": args.workers,
        "cpp_veclib_maximum_threads": args.workers,
        "warm_calls_per_block": 5,
        "timed_calls_per_block": args.calls,
        "blocks_per_runtime": args.blocks,
        "go_binary_sha256": sha256(args.go_binary),
        "cpp_binary_sha256": sha256(args.cpp_binary),
        "go_model_sha256": sha256(args.go_model),
        "cpp_model_sha256": sha256(args.cpp_model),
        "pcm_sha256": sha256(args.pcm),
        "go_blocks_ns": go_blocks,
        "cpp_blocks_ns": cpp_blocks,
        "go_median_ms": statistics.median(go) / 1e6,
        "cpp_median_ms": statistics.median(cpp) / 1e6,
        "median_ratio": statistics.median(go) / statistics.median(cpp),
        "ratio_95pct_block_bootstrap": ratio_interval(go_blocks, cpp_blocks),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
