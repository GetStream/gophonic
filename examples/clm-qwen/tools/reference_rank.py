#!/usr/bin/env python3
"""Regenerate TestOfficialCLMRanking's golden with Qwen3-8B BF16 and CLM heads.

Run: python tools/reference_rank.py /path/to/Qwen3-8B /path/to/CLM_v0.1-8B.pt
Requires torch and transformers, only for offline validation.
"""
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def project(vector, weights):
    x = F.gelu(F.linear(vector, weights["inp.weight"], weights["inp.bias"]))
    x = F.layer_norm(
        F.linear(x, weights["hidden.0.weight"], weights["hidden.0.bias"]),
        [1536], weights["norms.0.weight"], weights["norms.0.bias"],
    )
    x = F.gelu(x)
    return F.normalize(F.linear(x, weights["out.weight"], weights["out.bias"]), dim=-1)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: reference_rank.py QWEN3_DIR CLM_HEAD_PT")
    qwen_dir, head_file = sys.argv[1:]
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        qwen_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    ).eval()
    checkpoint = torch.load(head_file, map_location="cpu", weights_only=True)
    state = "What causes tides on Earth?"
    candidates = [
        "The Moon’s gravitational pull.",
        "Photosynthesis in plants.",
        "Because the Earth is round.",
    ]

    def embed(text):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return F.normalize(model(input_ids=ids).last_hidden_state[0, -1, :].float(), dim=-1)

    with torch.inference_mode():
        state_projection = project(embed(state), checkpoint["state_head"])
        actions = torch.stack([
            project(embed(candidate), checkpoint["action_head"])
            for candidate in candidates
        ])
        scale = torch.exp(checkpoint["logit_scale"]).clamp(max=100.0)
        probabilities = F.softmax(scale * actions @ state_projection, dim=0).tolist()
    for candidate, probability in sorted(zip(candidates, probabilities), key=lambda item: -item[1]):
        print(f"{probability:.12f}\t{candidate}")


if __name__ == "__main__":
    main()
