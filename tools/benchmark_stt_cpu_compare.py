#!/usr/bin/env python3
"""Alternate pinned Go test binaries with identical fixtures and CPU settings.

Build both revisions with project-env, GOEXPERIMENT=simd and CGO_ENABLED=0.
Run model oracle tests separately first. No profiling runs should overlap this
measurement. Results include every raw sample and binary hashes; intervals
resample paired blocks, not individual token timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--cwd", type=Path, required=True, help="package directory containing relative test fixtures")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.samples, args.iterations) < 1:
        p.error("samples and iterations must be positive")
    binaries = {name: str(path.resolve()) for name, path in (("before", args.before), ("after", args.after))}
    env = os.environ.copy()
    env["GOPHONIC_MODELS"] = str(args.models.resolve())
    report = {"benchmark": args.bench, "iterations": args.iterations, "binaries": {}, "samples": []}
    for name, path in binaries.items():
        report["binaries"][name] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    pattern = re.compile(r"^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op", re.M)
    for block in range(args.samples):
        for name in (("before", "after") if block % 2 == 0 else ("after", "before")):
            cmd = [binaries[name], "-test.run=^$", "-test.bench=" + args.bench,
                   f"-test.benchtime={args.iterations}x", "-test.count=1"]
            result = subprocess.run(cmd, cwd=args.cwd, env=env, text=True, capture_output=True, check=True)
            metrics = {m[1]: float(m[2]) for m in pattern.finditer(result.stdout)}
            if not metrics:
                raise RuntimeError("benchmark did not execute: " + result.stdout)
            report["samples"].append({"block": block, "variant": name, "ns_per_op": metrics, "raw": result.stdout})
            print(block, name, metrics, flush=True)
    rng = random.Random(0)
    summary = {}
    for bench in report["samples"][0]["ns_per_op"]:
        values = {name: [s["ns_per_op"][bench] for s in report["samples"] if s["variant"] == name] for name in binaries}
        pairs = list(zip(values["before"], values["after"]))
        ratios = []
        for _ in range(10000):
            draw = rng.choices(pairs, k=len(pairs))
            ratios.append(statistics.median(x[1] for x in draw) / statistics.median(x[0] for x in draw))
        ratios.sort()
        summary[bench] = {"before_median_ms": statistics.median(values["before"])/1e6,
                          "after_median_ms": statistics.median(values["after"])/1e6,
                          "after_over_before_95pct": [ratios[249], ratios[9749]]}
    report["summary"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
