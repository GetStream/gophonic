#!/usr/bin/env python3
# Copyright 2026 The gophonic authors
# SPDX-License-Identifier: BSD-2-Clause
"""Train qwen3asr's end-of-turn head (turn-1.7b.bin).

The head is a logistic regression over the decoder state that ends a
transcript. Training data is pipecat-ai/smart-turn-data-v3.2 (BSD-2-Clause):
speech cut at pauses, labeled whether the speaker's turn was over.

  1. prepare: download shards and write 16 kHz WAV clips and lists of
     (clip, label, pause in ms); each clip is judged after that much silence
     past the end of its speech, as a voice agent judges a pause.
       python3 turn.py prepare --out DIR --train-shards 4
  2. features: the Go test writes the decoder state for each clip.
       TURN_LIST=DIR/train.tsv TURN_OUT=DIR/train.bin go test ./qwen3asr -run TestTurnFeatures -timeout 3h
       TURN_LIST=DIR/test.tsv TURN_OUT=DIR/test.bin go test ./qwen3asr -run TestTurnFeatures
  3. train: fit, score on the held-out human speech, and write the head.
       python3 turn.py train --train DIR/train.bin --test DIR/test.bin --out ../turn-1.7b.bin

Needs numpy and scikit-learn; prepare also needs pyarrow and ffmpeg.
"""
import argparse
import os
import random
import struct
import subprocess
import sys
import urllib.request

import numpy as np

REPO = "https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-{split}/resolve/main/data/train-{i:05d}-of-{n:05d}.parquet"
SHARDS = {"train": 83, "test": 10}
PAUSES = [20, 40, 60, 100, 150, 200, 300]
HIDDEN, LAYERS, VOCAB = 2048, 28, 151936  # Qwen3-ASR-1.7B's decoder
MAGIC = b"gophonic turn 1\n"


def prepare(args):
    import pyarrow.parquet as pq

    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
    rng = random.Random(7)
    for split, shards in (("train", range(args.train_shards)), ("test", range(1))):
        lines = []
        for i in shards:
            path = os.path.join(args.out, f"{split}-{i:05d}.parquet")
            if not os.path.exists(path):
                urllib.request.urlretrieve(REPO.format(split=split, i=i, n=SHARDS[split]), path)
            t = pq.read_table(path, columns=["id", "language", "endpoint_bool", "synthetic", "audio"]).to_pydict()
            for j, clip in enumerate(t["id"]):
                # Held-out scoring uses human English only; training uses
                # everything.
                if split == "test" and (t["synthetic"][j] or t["language"][j] != "eng"):
                    continue
                wav = os.path.join(args.out, "clips", clip + ".wav")
                if not os.path.exists(wav):
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "pipe:", "-ac", "1", "-ar", "16000",
                                    "-c:a", "pcm_s16le", wav], input=t["audio"][j]["bytes"], check=True)
                pause = rng.choice(PAUSES) if split == "train" else 40
                lines.append(f"{wav}\t{int(t['endpoint_bool'][j])}\t{pause}\n")
        with open(os.path.join(args.out, f"{split}.tsv"), "w") as f:
            f.writelines(lines)
        print(f"{split}: {len(lines)} clips")


def load(path):
    rec = 3 + HIDDEN
    raw = np.fromfile(path, dtype="<f4")
    raw = raw[: len(raw) // rec * rec].reshape(-1, rec)
    return raw[:, 3 : 3 + HIDDEN].astype(np.float64), raw[:, 0] > 0.5


def score(name, p, y, threshold):
    pred = p >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    print(f"{name:34s} accuracy {(pred == y).mean():.3f}  precision {tp / max(tp + fp, 1):.3f}  "
          f"recall {tp / y.sum():.3f}  false ends {fp}/{int((~y).sum())}")


def train(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    x, y = load(args.train)
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    best = max((cross_val_score(LogisticRegression(C=c, max_iter=3000), z, y, cv=5).mean(), c)
               for c in (0.001, 0.003, 0.01, 0.03, 0.1))
    print(f"train: {len(y)} clips, cross-validated accuracy {best[0]:.3f} at C={best[1]}")
    m = LogisticRegression(C=best[1], max_iter=3000).fit(z, y)
    # Fold the standardization into the weights: w·(x-mean)/std + b.
    w = m.coef_[0] / std
    b = m.intercept_[0] - (w * mean).sum()
    if args.test:
        xt, yt = load(args.test)
        p = 1 / (1 + np.exp(-(xt @ w + b)))
        for th in (0.5, 0.7, 0.8, 0.9):
            score(f"held-out human speech @{th}", p, yt, th)
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<3I2f", HIDDEN, LAYERS, VOCAB, args.threshold, b))
        f.write(np.asarray(w, dtype="<f4").tobytes())
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--train-shards", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--test")
    t.add_argument("--out", required=True)
    t.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    {"prepare": prepare, "train": train}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
