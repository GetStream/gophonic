#!/usr/bin/env python3
"""Analyze unfiltered paired continuation samples; requires only Python 3."""
import json
import math
import random
import statistics
import sys


def analyze(path):
    data = json.load(open(path))
    groups = {name: {} for name in ("recompute", "reuse")}
    for sample in data["samples"]:
        groups[sample["variant"]][sample["pair"]] = sample
    pairs = sorted(groups["recompute"])
    assert pairs == sorted(groups["reuse"])
    rng = random.Random(250925)
    result = {"source": path, "pairs": len(pairs), "bootstrap_draws": 30000,
              "seed": 250925, "method": "paired log ratios, percentile 95% CI; all samples retained",
              "results": []}
    for column in range(-1, len(data["frames"])):
        values = {}
        for name, samples in groups.items():
            values[name] = [samples[p]["ns"] if column < 0 else samples[p]["steps_ns"][column]
                            for p in pairs]
        logs = [math.log(b / a) for a, b in zip(values["recompute"], values["reuse"])]
        boot = sorted(math.exp(sum(rng.choices(logs, k=len(logs))) / len(logs))
                      for _ in range(30000))
        result["results"].append({
            "scope": "whole_trace" if column < 0 else f"frames_{data['frames'][column]}",
            "ratio": math.exp(statistics.mean(logs)),
            "ratio_ci95": [boot[750], boot[29249]],
            "recompute_median_ms": statistics.median(values["recompute"]) / 1e6,
            "reuse_median_ms": statistics.median(values["reuse"]) / 1e6,
        })
    return result


if __name__ == "__main__":
    json.dump([analyze(path) for path in sys.argv[1:]], sys.stdout, indent=2)
    print()
