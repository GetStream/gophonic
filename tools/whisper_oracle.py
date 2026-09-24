#!/usr/bin/env python3
"""Generate pinned official-PyTorch tiny.en stage fixtures from JFK PCM.

Requires the OpenAI Whisper source at commit 86098128c0b4f24f0e2aa2994de830614b474227,
PyTorch, and the official SHA-pinned tiny.en checkpoint. This is a development
oracle only; the Go runtime has no Python or PyTorch dependency.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_SHA256 = "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
SOURCE_COMMIT = "86098128c0b4f24f0e2aa2994de830614b474227"
PCM_SHA256 = "80a6b1e2dcc00e55e5341e4de9c44f65bc576202c93106ff6b4bfe251a87e02f"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_f32(path, tensor):
    tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("openai_source")
    parser.add_argument("pcm", help="16 kHz mono float32 raw PCM")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    source = Path(args.openai_source)
    import subprocess
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise SystemExit(f"wrong OpenAI source commit {commit}")
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("wrong checkpoint SHA-256")
    if sha256(args.pcm) != PCM_SHA256:
        raise SystemExit("wrong PCM SHA-256")
    sys.path.insert(0, str(source))
    import whisper
    from whisper.model import ModelDimensions, Whisper

    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Whisper(ModelDimensions(**checkpoint["dims"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    pcm = np.fromfile(args.pcm, dtype="<f4")
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        audio = whisper.pad_or_trim(torch.from_numpy(pcm), length=whisper.audio.N_SAMPLES)
        mel = whisper.log_mel_spectrogram(audio)
        write_f32(dst / "jfk.mel.f32le", mel)
        stages = {}
        hooks = []
        for name, module in (
            ("conv1", model.encoder.conv1),
            ("conv2", model.encoder.conv2),
            *[(f"block{i}", block) for i, block in enumerate(model.encoder.blocks)],
        ):
            hooks.append(module.register_forward_hook(lambda _m, _in, out, key=name: stages.__setitem__(key, out)))
        encoded = model.encoder(mel.unsqueeze(0))
        for hook in hooks:
            hook.remove()
        write_f32(dst / "jfk.encoder.f32le", encoded[0])
        for name, value in stages.items():
            # Go's stem trace is time-major; PyTorch Conv1d returns [C,T].
            stage = value[0].transpose(0, 1) if name in ("conv1", "conv2") else value[0]
            write_f32(dst / f"jfk.encoder.{name}.f32le", stage)
        prompt = torch.tensor([[50257, 50362]], dtype=torch.long)
        prefix_logits = model.decoder(prompt, encoded)[0, -1]
        write_f32(dst / "jfk.prefix_logits.f32le", prefix_logits)
        after_first_logits = model.decoder(torch.tensor([[50257, 50362, 843]]), encoded)[0, -1]
        write_f32(dst / "jfk.after_843_logits.f32le", after_first_logits)
        half_audio_logits = model.decoder(prompt, encoded * 0.5)[0, -1]
        write_f32(dst / "jfk.half_audio_prefix_logits.f32le", half_audio_logits)
        result = whisper.decode(model, mel, whisper.DecodingOptions(
            language="en", task="transcribe", without_timestamps=True,
            fp16=False, temperature=0.0, beam_size=None, best_of=None,
        ))
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "pcm_sha256": PCM_SHA256,
        "mel_shape": [80, 3000],
        "encoder_shape": [1500, 384],
        "prefix": [50257, 50362],
        "prefix_argmax": int(torch.argmax(prefix_logits)),
        "transcript": result.text,
        "tokens": result.tokens,
        "files_sha256": {p.name: sha256(p) for p in sorted(dst.glob("*.f32le"))},
    }
    (dst / "jfk.oracle.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Full-file transcription uses a global mel transform of PCM plus thirty
    # seconds of right padding, followed by mel-space windowing. It can differ
    # from the single PCM-padded decode above on the same short clip.
    full_cases = []
    full_feature_values = {}
    inputs = (
        ("jfk", pcm, "../whisper_jfk.pcm.f32le"),
        ("silence_1s", np.zeros(16000, dtype=np.float32), "16000 zero-valued float32le mono PCM samples"),
        ("jfk_plus_35s_silence", np.concatenate((pcm, np.zeros(35 * 16000, dtype=np.float32))),
         "../whisper_jfk.pcm.f32le followed by 560000 zero-valued float32le mono PCM samples"),
    )
    with torch.inference_mode():
        for name, samples, description in inputs:
            full_mel = whisper.log_mel_spectrogram(torch.from_numpy(samples), padding=whisper.audio.N_SAMPLES)
            transcribed = model.transcribe(
                samples, temperature=0.0, without_timestamps=True,
                fp16=False, verbose=False,
            )
            segments = [
                {key: segment[key] for key in ("start", "end", "text", "tokens")}
                for segment in transcribed["segments"]
            ]
            final_tokens = [token for segment in segments for token in segment["tokens"]]
            record = {
                "name": name,
                "pcm": description,
                "pcm_sha256": hashlib.sha256(samples.astype("<f4", copy=False).tobytes()).hexdigest(),
                "pcm_samples": int(samples.size),
                "content_frames": int(samples.size // 160),
                "transcript": transcribed["text"],
                "final_tokens": final_tokens,
                "segments": segments,
                "segment_count": len(segments),
            }
            if name != "silence_1s":
                frames_to_check = (
                    [0, 100, 250, 500, 750, 1000, 1099, 1100, 1500, 2999, 3000, 4099]
                    if name == "jfk" else
                    [0, 100, 250, 500, 750, 1000, 1099, 2999, 3000, 3001, 3500, 4599, 4600, 6000, 7599]
                )
                full_feature_values[name] = {
                    "pcm_samples": int(samples.size),
                    "frames": int(full_mel.shape[-1]),
                    "samples": [
                        {"mel": mel, "frame": frame, "value": float(full_mel[mel, frame])}
                        for mel in (0, 1, 10, 40, 79)
                        for frame in frames_to_check
                    ],
                }
            if name != "jfk_plus_35s_silence":
                segment_mel = whisper.pad_or_trim(full_mel[:, : min(whisper.audio.N_FRAMES, record["content_frames"])])
                decoded = whisper.decode(model, segment_mel, whisper.DecodingOptions(
                    language="en", task="transcribe", without_timestamps=True,
                    fp16=False, temperature=0.0, beam_size=None, best_of=None,
                ))
                record.update({
                    "decode_text": decoded.text,
                    "tokens": decoded.tokens,
                    "no_speech_prob": decoded.no_speech_prob,
                    "avg_logprob": decoded.avg_logprob,
                })
            full_cases.append(record)
    full_manifest = {
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "oracle": "OpenAI Whisper transcribe.py at pinned source commit; CPU FP32 tiny.en",
        "options": {
            "temperature": 0.0,
            "without_timestamps": True,
            "fp16": False,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
        },
        "cases": full_cases,
    }
    (dst / "fullfile.greedy_notimestamps.oracle.json").write_text(json.dumps(full_manifest, indent=2) + "\n")
    feature_manifest = {
        "source_commit": SOURCE_COMMIT,
        "padding_samples": whisper.audio.N_SAMPLES,
        "sample_rate": whisper.audio.SAMPLE_RATE,
        "values": full_feature_values,
    }
    (dst / "full_features.oracle.json").write_text(json.dumps(feature_manifest, indent=2) + "\n")
    print(result.text)


if __name__ == "__main__":
    main()
