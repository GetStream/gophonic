#!/usr/bin/env python3
"""Record matched CPU-only whisper.cpp tiny.en runs on the JFK oracle audio.

Build whisper.cpp commit a664346ea5c6dddff3e61a2b7b32dd4514613f50
with -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF. Supply its
whisper-cli binary, the converted FP32 ggml model, and 16 kHz mono WAV.
"""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path


TIMING = re.compile(r"whisper_print_timings:\s+(.+?) time\s+=\s+([0-9.]+) ms")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--oracle", type=Path, default=Path("testdata/whisper/jfk.oracle.json"))
    args = parser.parse_args()
    if args.threads < 1 or args.runs < 1:
        parser.error("threads and runs must be positive")
    checkout = args.cli.resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != "a664346ea5c6dddff3e61a2b7b32dd4514613f50":
        raise RuntimeError(f"unexpected whisper.cpp commit {commit}")
    cache = (args.cli.resolve().parents[1] / "CMakeCache.txt").read_text()
    flags = {"GGML_METAL": "OFF", "GGML_ACCELERATE": "OFF", "GGML_BLAS": "OFF", "WHISPER_COREML": "OFF"}
    for name, value in flags.items():
        if not re.search(rf"^{name}:BOOL={value}$", cache, flags=re.MULTILINE):
            raise RuntimeError(f"whisper.cpp build does not verify {name}={value}")
    oracle = json.loads(args.oracle.read_text())
    expected_transcript = oracle["transcript"].strip()
    command = [str(args.cli), "-m", str(args.model), "-f", str(args.wav),
               "-l", "en", "-t", str(args.threads), "-bs", "1", "-bo", "1", "-nf", "-nt", "-ng"]
    runs = []
    for _ in range(args.runs):
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        if "whisper_backend_init_gpu: no GPU found" not in output:
            raise RuntimeError("CPU-only GPU check absent from whisper.cpp output")
        values = {name.strip(): float(ms) for name, ms in TIMING.findall(output)}
        if not {"mel", "encode", "decode", "total"}.issubset(values):
            raise RuntimeError("whisper.cpp timing fields missing")
        if result.stdout.strip() != expected_transcript:
            raise RuntimeError("whisper.cpp transcript differs from JFK oracle")
        runs.append(values)
    report = {
        "whisper_cpp_commit": commit,
        "cmake_flags": flags,
        "model_sha256": sha256(args.model),
        "wav_sha256": sha256(args.wav),
        "command": command,
        "transcript": expected_transcript,
        "runs_ms": runs,
        "median_ms": {key: statistics.median(row[key] for row in runs) for key in runs[0]},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
