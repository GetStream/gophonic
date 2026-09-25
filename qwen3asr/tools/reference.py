#!/usr/bin/env python3
"""Write the Qwen3-ASR reference that qwen3asr's tests compare against.

Usage: reference.py MODEL_DIR OUT_DIR

Runs the official qwen-asr package (github.com/QwenLM/Qwen3-ASR) in FP32 on
the CPU for each test clip and writes OUT_DIR/reference.json plus a few
encoder rows per clip as little-endian float32. The clips are the 16 kHz
PCM files in testdata: whisper_jfk.pcm.f32le, peak-normalized as the
package's normalize_audio_input does, and qwen3asr/zh.pcm.s16le.

The audio encoder attends within windows of n_window_infer frames. vLLM and
flash attention pass those windows as cu_seqlens, but the Transformers
encoder's eager and SDPA paths drop them and attend across the whole clip,
so this script supplies the windows as a block-diagonal mask, as the
windowed backends compute.
"""
import json
import os
import sys

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import modeling_qwen3_asr as mq
from qwen_asr.inference.utils import parse_asr_output

model_dir, out_dir = sys.argv[1], sys.argv[2]
testdata = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")

layer_forward = mq.Qwen3ASRAudioEncoderLayer.forward


def windowed(self, hidden_states, cu_seqlens, attention_mask=None, **kw):
    n = hidden_states.shape[0]
    mask = torch.full([1, 1, n, n], torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
    for i in range(1, len(cu_seqlens)):
        a, b = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
        mask[..., a:b, a:b] = 0
    return layer_forward(self, hidden_states, cu_seqlens, attention_mask=mask, **kw)


mq.Qwen3ASRAudioEncoderLayer.forward = windowed


def clip_jfk():
    wav = np.fromfile(os.path.join(testdata, "whisper_jfk.pcm.f32le"), dtype="<f4")
    return np.clip(wav / float(np.max(np.abs(wav))), -1, 1).astype(np.float32)


def clip_zh():
    return (np.fromfile(os.path.join(testdata, "qwen3asr", "zh.pcm.s16le"), dtype="<i2") / 32768).astype(np.float32)


asr = Qwen3ASRModel.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu", max_new_tokens=512)
thinker = asr.model.thinker
revision = None  # the Hugging Face commit, when hf download recorded it
meta = os.path.join(model_dir, ".cache", "huggingface", "download", "config.json.metadata")
if os.path.exists(meta):
    revision = open(meta).readline().strip()
out = {"model": "Qwen/Qwen3-ASR-1.7B", "revision": revision, "torch": torch.__version__,
       "dtype": "float32", "encoder_attention": "windowed", "clips": {}}
for name, wav in (("jfk", clip_jfk()), ("zh", clip_zh())):
    prompt = asr._build_text_prompt(context="", force_language=None)
    inputs = asr.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    frames = int(inputs["feature_attention_mask"][0].sum())
    feats = inputs["input_features"][0][:, :frames]
    with torch.no_grad():
        enc = thinker.get_audio_features(inputs["input_features"], feature_attention_mask=inputs["feature_attention_mask"])
        logits = thinker(**inputs).logits[0, -1]
        gen = asr.model.generate(**inputs, max_new_tokens=512).sequences[0, inputs["input_ids"].shape[1]:].tolist()
    raw = asr.processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    language, text = parse_asr_output(raw)
    # Rows at both ends of each attention window.
    rows = sorted({0, enc.shape[0] - 1} | {r for w in range(104, enc.shape[0], 104) for r in (w - 1, w)})
    enc[rows].float().numpy().astype("<f4").tofile(os.path.join(out_dir, name + ".encoder.f32le"))
    frame_idx = [0, 1, frames // 2, frames - 1]
    top = torch.topk(logits, 32)
    out["clips"][name] = {
        "samples": len(wav), "frames": frames, "tokens": enc.shape[0],
        "feature_frames": frame_idx, "features": feats[:, frame_idx].T.flatten().tolist(),
        "encoder_rows": rows, "input_ids": inputs["input_ids"][0].tolist(),
        "logits_top": [[int(i), float(v)] for v, i in zip(top.values, top.indices)],
        "generated": gen, "raw": raw, "language": language, "text": text,
    }
    print(name, frames, enc.shape, repr(raw))
with open(os.path.join(out_dir, "reference.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
