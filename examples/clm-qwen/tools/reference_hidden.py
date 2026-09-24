#!/usr/bin/env python3
"""Write the official BF16 Qwen3 last hidden state of a text as float32.

Usage: reference_hidden.py MODEL_DIR OUTPUT.f32 [TEXT]

The gated Go test TestOfficialHelloMatchesBF16Reference compares against this
vector (TEXT defaults to "hello"). No special tokens are added, matching CLM.
"""
import sys

import torch
from transformers import AutoModel, AutoTokenizer

path, out = sys.argv[1], sys.argv[2]
text = sys.argv[3] if len(sys.argv) > 3 else "hello"
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True).eval()
ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
with torch.inference_mode():
    hidden = model(input_ids=ids).last_hidden_state[0, -1, :].float().cpu().numpy()
hidden.astype("<f4").tofile(out)
print("tokens", ids.tolist(), "norm", float((hidden**2).sum() ** 0.5))
